import chainlit as cl
from chainlit.input_widget import Switch
import asyncio
import json
import torch

from functools import lru_cache
import re
import sys
import os
import datasets
import outlines
import logging
import frontmatter
import utils.preprocessing as pre
import shutil
import utils.json_schema as sch
import networkx as nx
import ahocorasick
import string
import ast

from gliner import GLiNER
import utils.prompts as prompts
from utils.json_utils import JSONUtils
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,TorchAoConfig,AutoTokenizer,BartTokenizer, BartForConditionalGeneration
from pydantic import BaseModel, Field
from datasets import Dataset,concatenate_datasets
from typing import Literal
from sentence_transformers import SentenceTransformer
from utils.semantic import Semantic
from json_repair import repair_json
from pathlib import Path
from utils.io_utils import IO_Utils
from utils.pydantic_schema import ReasonedResponse,SceneSummary

def uppercase(text):
    return text[0].upper()+text[1:]

# Used by the response parser for tool
class RAGQuerySchema(BaseModel):
    query: str = Field(description="The search query to lookup local markdown files or notes.")

helpers = [{"id":"Ideate", "icon":"lightbulb", "description":"Add ideas to knowledge base"},
                {"id":"Analyze", "icon":"brain", "description":"Analyze uploaded text"},
                {"id":"Update", "icon":"list-restart", "description":"Update internal knowledge base"},
                {"id":"Check", "icon":"list-checks", "description":"Check an idea"},
                {"id":"Metadata", "icon":"tag", "description":"Add Metadata"}]


with open('config.json', "r", encoding="utf-8") as f:
        # Load the JSON data into a Python dictionary
        config = json.load(f)
store_path= config['data_dir']


io_utils = IO_Utils()

model_id = "meta-llama/Llama-3.2-3B-Instruct"

named_entities = ['character','location','artifact','faction','event','definition']

#"Qwen/Qwen2.5-0.5B-Instruct" #
punctuation_tuple = tuple(string.punctuation)
persistent_actions = [
]

#@lru_cache(maxsize=32)
def get_or_create_generator(model, schema_class):
  """Caches the Outlines JSON generator so the FSM isn't recompiled

  on every request, preventing token desync errors.
  """
  outline_model = outlines.from_transformers(model, tokenizer)

  return outlines.Generator(outline_model,schema_class)

@cl.cache
def load_extraction_model():
    extraction_model_id = "urchade/gliner_small-v2.1"
    extraction_model = GLiNER.from_pretrained(extraction_model_id)

    return extraction_model

@cl.cache
def load_models():    

    '''hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            ignore_mismatched_sizes=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager",
        )'''
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    print("Wrapping model with Outlines (v0.3+ style)...")
    model = outlines.from_transformers(hf_model, tokenizer)
    #english_regex = r"[\x20-\x7E\n]+"
    generator_model = outlines.Generator(model)
    #model = outlines.generate.regex(hf_modela, english_regex)

    return model, tokenizer, generator_model,hf_model

@cl.cache
def load_embedding_models(): 
    # Load local FAISS and Embeddings
    STORE_DIR = f"{store_path}/FAISS_store/"
    #dataset = datasets.load_from_disk(os.path.join(STORE_DIR, "obsidian_dataset"))
    #dataset.load_faiss_index("embeddings", os.path.join(STORE_DIR, "obsidian_index.faiss"))
    dataset = datasets.load_from_disk(os.path.join(STORE_DIR, "worldbuilding_dataset"))
    dataset.load_faiss_index("embeddings", os.path.join(STORE_DIR, "worldbuilding_dataset.faiss"))
    embed_model = SentenceTransformer("BAAI/bge-m3")

    # To convert Q/A into Statements
    #qa_tokenizer = BartTokenizer.from_pretrained("MarkS/bart-base-qa2d")
    #qa_model = BartForConditionalGeneration.from_pretrained("MarkS/bart-base-qa2d")

    return embed_model, dataset


@cl.cache
def load_semantic_consistency_models(): 
    
    nli_model_id        = "cross-encoder/nli-deberta-v3-large"
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id)

    return nli_model,nli_tokenizer

generator_model, tokenizer, model, hf_model = load_models()
embed_model, dataset = load_embedding_models()
nli_model,nli_tokenizer = load_semantic_consistency_models()
extraction_model  = load_extraction_model()


sem = Semantic()

# Remove after verifying libraries.

sem.load_embedding_model(embed_model,dataset)
sem.load_nli_model(nli_model,nli_tokenizer)
sem.load_extraction_model(extraction_model)

json_utils = JSONUtils()
# TOOLS

available_tools = [{
    "name": "retrieve_local_notes",
    "description": "Looks up local context. Use this tool when the user context does not contain useful information to answer the query.",
    "input_schema": {
        "type": "object",
        "properties": {"query": {"type": "string"}, "threshold": {"type": "float"}, "k": {"type": "int"}},
        "required": ["query"]
    }
}]

async def retrieve_graph_rag(query,threshold=0.4,k=10,hops=1):

    graph = cl.user_session.get("knowledge_graph")
    documents_lookup = cl.user_session.get("documents_lookup")

    retrieved_ids = set()
    full_graph_context = dict()
    full_graph_context['direct'] = []
    full_graph_context['indirect'] = []

    
    retrieved_contexts,ids = sem.get_graph_rag_context(query,graph,documents_lookup,threshold,k,hops)

    for key in ['direct','indirect']:
        for idx in range(len(ids[key])):
            if(not(ids[key][idx] in retrieved_ids)):
                full_graph_context[key].append(retrieved_contexts[key][idx])
                retrieved_ids.add(ids[key][idx])

    # Format retrieved context
    full_context_text = ""
    if(len(full_graph_context['indirect'])>0):
        full_context_text+=f"SUPPLEMENTARY INFORMATION:\n\n{"\n\n".join(full_graph_context['indirect'])}\n\n"
    if(len(full_graph_context['direct'])>0):
        full_context_text += f"DIRECT CONTEXT:\n\n{"\n\n".join(full_graph_context['direct'])}"

    if(len(full_context_text)==0):
        full_context_text = "No context found."

    return  full_context_text,full_graph_context

# Legacy
async def retrieve_local_notes(query: str, threshold: float = 0.4, k = 10) -> str:

    loaded_dataset = cl.user_session.get("faiss_dataset")
    query_vector = embed_model.encode(query, normalize_embeddings=True)
    scores, examples = loaded_dataset.get_nearest_examples("embeddings", query_vector, k=k)
    
    matches = []

    for i in range(len(scores)):
        if scores[i] >= threshold:
            matches.append(f"{examples['text'][i]}")
        else:
            break
            
    return "\n\n".join(matches) if matches else "No matching local documents found.",matches

async def get_most_relevant_file(query: str, threshold: float = 0.4, k = 1) -> str:

    return sem.get_most_relevant_file(query,threshold,k)

#############
# Callbacks
#############

@cl.action_callback("show_ideas_history")
async def show_ideas_history():

    async with cl.Step(name="Show Idea History",icon="lightbulb") as step:

        messages = cl.chat_context.get()
        messages.reverse()
        for idx in range(len(messages)):
            if(not(messages[idx].author=='Assistant')):
                break
        messages = messages[:idx]

        for m in messages:
            print(m.author,m.content)

        for m in messages:
            await m.remove()

        chat_history = cl.user_session.get("idea_history")

        msgs = []
        for key in chat_history.keys():
            actions = [
                        cl.Action(
                            name="remove_from_idea_history",
                            icon="no",
                            payload={"key": key},
                            label="Remove"
                            ),
                        cl.Action(
                            name="crosscheck",
                            icon="no",
                            payload={"key": key, "idea":chat_history[key]},
                            label="Cross-Check"
                            )
                    ]
            msgs.append(cl.Message(content=chat_history[key],actions=actions))

        for msg in msgs:
            await msg.send()

@cl.action_callback("crosscheck")
async def crosscheck(action: cl.Action):

    async with cl.Step(name="Checking for Contradictions",icon='circle-question-mark') as step:

        idea = action.payload['idea']

        task_list = cl.user_session.get("task_list")
        for idx in range(len(task_list.tasks)):
            if(task_list.tasks[idx].forId==action.forId):    
                task_list.tasks[idx].status=cl.TaskStatus.RUNNING
                break

        await task_list.send()

        step.output = idea

        context_text, context_list = await retrieve_local_notes(idea)

        contradictory_lore = []
        if(len(context_list)>0):
            for i,context in enumerate(context_list):
                # Use only the textual part of the context for NLI to preserve tokens + not include tags etc.
                trimmed_context = [c for c in context.split('\n') if len(c)>0]
                contradiction_prob,_,_ = sem.get_entailment_probs(trimmed_context[-1],idea,nli_model,nli_tokenizer)
                if(contradiction_prob>0.5):
                    contradictory_lore.append([trimmed_context[-1],contradiction_prob])


            if(len(contradictory_lore)>0):
                await cl.Message(content='Found Potentially Contradictory Lore').send()
                for clore in contradictory_lore:
                    await cl.Message(content=clore[0]).send()

            else:
                await cl.Message(content='No Contradictory Lore Found!').send()

        else:
            await cl.Message(content='No Contradictory Lore Found!').send()


        for idx in range(len(task_list.tasks)):
            if(task_list.tasks[idx].forId==action.forId):
                if(len(contradictory_lore)>0):
                    task_list.tasks[idx].status=cl.TaskStatus.FAILED
                    break
                else:
                    task_list.tasks[idx].status=cl.TaskStatus.DONE
                    break

        await task_list.send()

@cl.action_callback("remove_from_idea_history")
async def remove_from_idea_history(action: cl.Action):

    task_list = cl.user_session.get("task_list")

    for idx in range(len(task_list.tasks)):
        if(task_list.tasks[idx].forId==action.forId):
            del task_list.tasks[idx]
            break

    await task_list.send()

    return True

@cl.action_callback("add_to_knowledge_base")
async def on_action(action: cl.Action):

    await save_idea_to_local_session(action.payload['idea'])

@cl.action_callback("show_reasoning")
async def show_reasoning(action: cl.action):
    await cl.Message(content=action.payload['content']).send()

@cl.action_callback("add_blurb")
async def add_blurb(action: cl.Action):

    actions = [
                cl.Action(
                    name="add_blurb",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Blurb"
                ),
                cl.Action(
                    name="add_alias",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Alias"
                ),
                cl.Action(
                    name="add_tag",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Tag"
                ),
            ]

    exists,data = json_utils.load_json(action.payload['topic'])
    if(len(data['blurb'].strip())>0):
        await cl.Message(content=f'Currently : {data['blurb']}').send()
    res = await cl.AskUserMessage(content=f"Provide a description of {uppercase(action.payload['topic'])} in common words.",timeout=60).send()
    if(res):    
        if(exists):
            data['blurb'] = res['output']
            json_utils.save_json(action.payload['topic'],data)
            await cl.Message(content=f'Saved blurb for {uppercase(action.payload['topic'])}',actions=actions).send()
        else:
            await cl.Message(content=f'File not found!').send()

    await cl.context.emitter.task_end()

@cl.action_callback("add_alias")
async def add_alias(action: cl.Action):
    actions = [
                cl.Action(
                    name="add_blurb",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Blurb"
                ),
                cl.Action(
                    name="add_alias",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Alias"
                ),
                cl.Action(
                    name="add_tag",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Tag"
                ),
            ]

    exists,data = json_utils.load_json(action.payload['topic'])
    if(len(data['aliases'])>0):
        await cl.Message(content=f'Currently : {','.join(data['aliases'])}').send()
    res = await cl.AskUserMessage(content=f"Provide an alias for {uppercase(action.payload['topic'])}.",timeout=60).send()

    if(res):
        if(exists):
            aliases = res['output'].split(',')
            data['aliases'] += aliases
            data['aliases'] = list(set(data['aliases']))
            json_utils.save_json(action.payload['topic'],data)
            await cl.Message(content=f'Saved aliases for {uppercase(action.payload['topic'])}',actions=actions).send()
        else:
            await cl.Message(content=f'File not found!').send()

    await cl.context.emitter.task_end()

@cl.action_callback("add_tag")
async def add_tag(action: cl.Action):

    actions = [
                cl.Action(
                    name="add_blurb",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Blurb"
                ),
                cl.Action(
                    name="add_alias",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Alias"
                ),
                cl.Action(
                    name="add_tag",
                    icon="",
                    payload={'topic':action.payload['topic']},
                    label="Add Tag"
                ),
            ]

    exists,data = json_utils.load_json(action.payload['topic'])
    if(len(data['tags'])>0):
        await cl.Message(content=f'Currently : {','.join(data['tags'])}').send()
    res = await cl.AskUserMessage(content=f"Provide tags for {uppercase(action.payload['topic'])}.",timeout=60).send()

    if(res):
        if(exists):
            aliases = res['output'].split(',')
            data['tags'] += aliases
            json_utils.save_json(action.payload['topic'],data)
            await cl.Message(content=f'Saved tags for {uppercase(action.payload['topic'])}',actions=actions).send()
        else:
            await cl.Message(content=f'File not found!').send()

    await cl.context.emitter.task_end()
        

#############
# Tools
#############


async def split_scenes(file):

    #file = user_message.elements[0]
    sentences = io_utils.load_text_sentences(file.path)

    scenes = []
    curr_scene = []
    window_size = 2
    start_idx = 0
    end_idx = 0

    chunks = []
    for idx in range(len(sentences)-window_size):
        chunk = await add_entity_context(' '.join(sentences[idx:idx+window_size]))
        chunks.append(chunk)

    chunk_embeddings = sem.embed_text(chunks)

    for idx in range(len(chunk_embeddings)-1):
        cosine_sim = sem.get_cosine_similarity(chunk_embeddings[idx],chunk_embeddings[idx+1])
        #print(s1)
        #print(s2)
        #print(cosine_sim)
        #print()

        end_idx = idx+window_size+1
        if(cosine_sim<0.75):
            scene = await add_entity_context(' '.join(sentences[start_idx:end_idx+1]))
            scenes.append(scene)
            start_idx = idx+window_size+1
            
    
    end_idx = len(sentences)
    scene = await add_entity_context(' '.join(sentences[start_idx:end_idx]))
    scenes.append(scene)
    
    return scenes

async def get_gist(user_message):

    file = user_message.elements[0]
    #text_rows = io_utils.load_text_rows(file.path)
    scenes = await split_scenes(file)

    summary = 'No context available.' 

    for scene in scenes:
        #scene_with_context = await add_entity_context(scene)
        prompt = prompts.get_gist_prompt(text=scene,context=summary, subjects = named_entities[:-1]) # Ignore 'definitions'
        print(prompt)
        summary = await tokenize_and_generate(prompt,max_new_tokens=256,temperature=0.3,template=SceneSummary)
        scene_summary = SceneSummary.model_validate_json(summary)
        scene_summary = scene_summary.model_dump()
        print(scene_summary)
        print()

        await cl.Message(content=scene_summary).send()

        # Redesign the summary to feed back as context.
        summary = dict()
        for key in scene_summary.keys():
            summary[key] = ', '.join(scene_summary[key])

        '''for role in scene_summary.keys():

            try:
                print(role)
                # Dynamically create Pydantic schema for role -> schema, keys
                if(role.endswith('s')): role = role[:-1]
                RoleSchema = sch.get_pydantic_schema('RoleSchema',role)

                # Pass keys, role and entity to prompt creation function
                role_subjects = list(RoleSchema.model_fields.keys())

                # Pass pydantic schema to tokenize_and_generate
                for entity in scene_summary[role]:
                    scene_extraction_prompt = prompts.isolate_scene_element(text=scene,entity=entity,aspect=role,subjects=role_subjects)
                    print(scene_extraction_prompt)
                    isolated_element = await tokenize_and_generate(scene_extraction_prompt,max_new_tokens=256,temperature=0.6,template=RoleSchema)
                    await cl.Message(content=isolated_element).send()
                    print(isolated_element)
                    print()
                    print()
            except Exception as e:
                print(e)'''

async def extract_entities(user_message):

    tracked_entities = cl.user_session.get("entities_to_track")
    file = user_message.elements[0]

    entities = sem.named_entity_extraction(file.path)

    items_list = []
    items_dict = dict()
    for entity,label in entities:

        items_list.append(
                {
                    "id":entity.lower(),
                    "label":entity,
                    "description":label,
                    "defaultChecked": False
                })

        items_dict[entity.lower()] = label

    print(items_dict)
    props = {
            "timeout": 6000,
            "topText": "Found Named Entities in Text",
            "Title": "Select Entities to Add Context",
            "items": items_list}

    checklist_element = cl.CustomElement(
        name="SelectToTrack",
        props=props
    )

    # 3. Send the component attached to a chat message
    selection_response = await cl.AskElementMessage(
        content="",
        element=checklist_element
    ).send()

    print(selection_response)

    for key in selection_response.keys():
        if(not(key=='submitted') and selection_response[key]):

            entity = key
            label = items_dict[key]

            response = await cl.AskActionMessage(
                content=f"What is {uppercase(entity)}?",
                actions=[cl.Action(name=n, payload={"label": n}, label=n[0].upper()+n[1:]) for n in named_entities]).send()

            label = response['payload']['label']

            tracked_entities.append(entity)

            # Check if the entity already exists in the JSON store

            filename = entity.lower()
            filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            if(Path(f"{store_path}/{filename}.json").exists()):
                with open(f"{store_path}/{filename}.json", "r") as file:
                    data = json.load(file)
                schema = data['data']
            else:
                data = dict()
                data['name'] = entity
                data['type'] = label
                schema = sch.get_schema(label)

            print(schema)

            # Prompt the user with the form element and wait for response

            response = await create_schema_element(entity,label,schema)

            if(response['submitted']):
                for key in schema.keys():
                    if(response[key].strip()):
                        schema[key].append(response[key])

            data['data'] = schema
            data['tags'] = []
            data['related'] = []

            # 2. Write the dictionary directly to a JSON file
            filename = entity.lower()
            filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            with open(f"{store_path}/{filename}.json", "w") as file:
                json.dump(data, file, indent=4)

            #await delete_last_message()


        #await delete_last_message()

    cl.user_session.set("entities_to_track",tracked_entities)

    await cl.Message(f"Tracking {', '.join([uppercase(d) for d in cl.user_session.get("entities_to_track")])}...").send()

async def create_schema_element(entity,label,schema,message="Please describe your idea!",initial_tab='overview'):

    print(entity)
    print(label)
    print(schema)

    props = {
            "timeout": 6000,
            "initialTab":initial_tab,
            "enableEdit": True,
            "topText": label[:1].upper()+label[1:],
            "Title": entity[:1].upper()+entity[1:],
            "fields": []}

    for key in schema.keys():
        new_field = dict()
        new_field['id'] = key
        new_field['label'] = key[:1].upper()+key[1:]
        new_field['type'] = 'text'
        new_field['value'] = ''
        new_field['description'] = '. '.join(schema[key])
        props['fields'].append(new_field)

    element = cl.CustomElement(
                    name="KnowledgeBase",
                    display="inline",
                    props=props
                )

    print(element)

    response = await cl.AskElementMessage(
                content=message,
                element=element,
                timeout=6000
            ).send()

    return response

async def create_knowledge_graph():

    graph,documents_lookup,filename_to_name = json_utils.create_knowledge_graph()

    cl.user_session.set('knowledge_graph',graph)
    cl.user_session.set('documents_lookup',documents_lookup)

async def update_internal_knowledge_base():

    documents = await cl.make_async(json_utils.load_files)()

    new_dataset,index = await cl.make_async(sem.create_new_dataset)(documents)

    await cl.make_async(json_utils.overwrite_faiss_dataset)(new_dataset,index)

    await cl.make_async(sem.reload_faiss_dataset)()

async def delete_last_message():
    chat_context = cl.chat_context.get()
    await chat_context[-1].remove()

async def update_faiss():

    dataset_dir = '/Users/satyawagle/Projects/LangChain/llm_chatbot/data/json_store'

    async with cl.Step(name=f"Updating Knowledge Database") as step:

        json_utils.update_links()

        await create_knowledge_graph()
        
        await update_internal_knowledge_base()

async def add_entity_context(idea,entity_threshold=0.4):

    blurb_map = cl.user_session.get('blurb_map')
    alias_map = cl.user_session.get('alias_map')

    '''entities = sem.entity_extraction(idea,entity_threshold=entity_threshold)
    print(entities)
    entity_context = dict()
    for entity in entities:
        exists,data = json_utils.load_json(entity)
        if(exists):
            entity_context[uppercase(entity)] = (''.join(data['blurb'])).lower()'''

    entity_context = dict()

    if(idea.endswith(punctuation_tuple)):
        idea = idea[:-1]

    for og_word in idea.split(' '):

        if(og_word.lower() in alias_map.keys()):
            word = alias_map[og_word.lower()]
        else:
            word = og_word

        if(word.lower() in blurb_map.keys()):
            blurb = blurb_map[word.lower()]

            entity_context[og_word] = blurb

    for key in entity_context.keys():
        if(entity_context[key].endswith('.')):
            entity_context[key] = entity_context[key][:-1]

        if(len(entity_context[key])>0):
            idea = idea.replace(key,f"{key} ({entity_context[key]})")

    return idea


#############
# Core Functions
#############

async def tokenize_and_generate(chat_history,max_new_tokens=256,temperature=0.6,template=None):

    synthesis_prompt = tokenizer.apply_chat_template(
                                    chat_history,
                                    tokenize=False,
                                    add_generation_prompt=True
                                    )

    if(template):
        generator = get_or_create_generator(hf_model, template)
        answer = await cl.make_async(generator)(
        synthesis_prompt,
        max_new_tokens=max_new_tokens, 
        temperature=temperature
        )
    else:
        answer = await cl.make_async(generator_model)(
        synthesis_prompt,
        max_new_tokens=max_new_tokens, 
        temperature=temperature
        )

    return answer


async def query_processing(user_input,entity_threshold=0.4):

    print(user_input)

    # Replace aliases
    alias_map = cl.user_session.get('alias_map')
    for key in alias_map.keys():
        user_input = user_input.replace(key,alias_map[key])

    user_input_with_context = await add_entity_context(user_input,entity_threshold)

    print(user_input_with_context)

    return user_input_with_context

@cl.step(name='Retrieve Context')
async def retrieve_context(user_input):

    #chat_history = []
    #chat_history = prompts.get_context_retrieval_prompt('system',chat_history)
    #chat_history = prompts.get_context_retrieval_prompt('user',chat_history,user_input,', '.join(relevant_entities))

    # Construct Tool Call
    relevant_entities = sem.entity_extraction(user_input,entity_threshold=0.25)

    async with cl.Step(name="Retrieve Context") as step:

        # Context Retrievant Via MCP
        user_input_with_context = await query_processing(user_input)
        context_text, context_list = await retrieve_graph_rag(user_input_with_context)

        print(context_text)

    return context_text,context_list

@cl.step(name='Check Context Sufficiency')
async def check_context_sufficiency(proposition,context_dict):

    # Check if Context can answer the Question

    max_prob = 0.0
    best_context = 0

    context_list = []
    for key in context_dict.keys():
        context_list += context_dict[key]

    #proposition = await add_entity_context(proposition)
    print(proposition)

    if(len(context_list)>0):

        for i,context in enumerate(context_list):
            # Use only the textual part of the context for NLI to preserve tokens + not include tags etc.
            trimmed_context = [c for c in context.split(':') if len(c)>0]
            trimmed_context = trimmed_context[-1].strip()
            #trimmed_context = await add_entity_context(trimmed_context)
            print(trimmed_context)
            contradiction,entailment_probs,neutral = sem.get_entailment_probs(trimmed_context,proposition)
            print(contradiction,entailment_probs,neutral)
            print()
            #print(context,entailment_probs)
            if(entailment_probs>max_prob):
                max_prob = entailment_probs
                best_context = i

        #print(f"Best Context with Entailment Probability {max_prob}")
        #print(context_list[best_context])

    return best_context, max_prob

@cl.step(name='Identify Additional Context')
async def identify_additional_context(user_input,context_text):
        synthesis_history = [
                            {"role": "system", "content": f"You are a helpful writing assistant. You must answer with at least 3 and at most 5 additional questions that will help answer the user query with local context. You cannot ask the user query as a question. Your response must be in the form of bullet points using the bullet marker '*'"},
                            {"role": "user", "content": f"User Query: {user_input}\n\nLocal Context :\n{context_text}"},
                            ]

        additional_queries = await tokenize_and_generate(synthesis_history,temperature=0.3)
        questions = re.findall(r'\*\s([^\n]+)',additional_queries)

        is_answered = False
        for question in questions:
            context_text,answered = await get_context_from_user(question,context_text)
            is_answered = is_answered or answered
            print(answered,is_answered)


        return context_text,is_answered

async def get_context_from_user(question,context_text):

    res = await cl.AskActionMessage(
                content=question,
                actions=[
                    cl.Action(name="answer", payload={"value": "answer"}, label="Answer"),
                    cl.Action(name="skip", payload={"value": "skip"}, label="Skip"),],
                    ).send()

    answered = False
    if res and res.get("payload").get("value") == "answer":
        answer = await cl.AskUserMessage(content=q, timeout=120).send()

        await cl.Message(content=answer['output'],actions=persistent_actions).send()
        context_text += f' {answer['output']}'

        # Sending an action button within a chatbot message (NOT NEEDED)
        actions = [
            cl.Action(
                name="add_to_knowledge_base",
                icon="plus-sign",
                payload={"idea":answer['output']},
                label="Add to Knowledge Base"
            )
        ]

        await delete_last_message()

        answered = True

    return context_text,answered

async def check_idea(user_input):

    exists,data = json_utils.load_json(user_input)

    if(exists):

        entity = data['name']
        schema = data['data']
        label = data['type']
        
        props = {
                "timeout": 6000,
                "initialTab":'overview',
                "enableEdit": True,
                "initEdit": False,
                "topText": label[:1].upper()+label[1:],
                "Title": entity[:1].upper()+entity[1:],
                "fields": []}

        for key in schema.keys():
            new_field = dict()
            new_field['id'] = key
            new_field['label'] = key[:1].upper()+key[1:]
            new_field['type'] = 'text'
            new_field['value'] = ''
            new_field['description'] = schema[key]#' '.join(schema[key])
            props['fields'].append(new_field)

        element = cl.CustomElement(
                        name="KnowledgeBase",
                        display="inline",
                        props=props
                    )

        response = await cl.AskElementMessage(
                    content=f"",
                    element=element,
                    timeout=6000
                ).send()

        # TODO : SAVE JSON to filesystem

        if(response['submitted']):
            for key in schema.keys():
                if(len(response[key])>0):
                    #schema[key].append(response[key])
                    schema[key] = [x for x in response[key] if x.strip()]

            data['data'] = schema
            data['tags'] = []
            data['related'] = dict()

            for key in data['data'].keys():
                data['related'][key] = []

            filename = entity.lower()
            filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            with open(f"{store_path}/json_store/{filename}.json", "w") as file:
                json.dump(data, file, indent=4)

            await cl.Message(content=f'Updated information about {uppercase(entity)}!').send()

    else:
        topic_response = await cl.Message(content="Topic not found!").send()

async def check_idea_for_contradictions(idea:str):

    _,context_dict = await retrieve_context(idea)

    _,max_prob = await check_context_sufficiency(idea,context_dict)

async def save_idea_to_local_session(idea:str):    

    idea_actions = [
        cl.Action(
            name="crosscheck",
            icon="circle-question-mark",
            payload={"idea": idea},
            label="Cross Check Idea"
        ),
        cl.Action(
            name="remove_from_idea_history",
            icon="trash-2",
            payload={"none": " "},
            label="Remove Idea"
        )
    ]

    cancel_action = [
        cl.Action(
            name="cancel_button",
            icon="trash-2",
            payload={"value": True},
            label="Click me!"
        )
    ]

    # Add punctuation if it isn't there.
    if(not(idea.endswith('.'))):
        idea += '.'

    most_relevant_file = await get_most_relevant_file(idea,threshold=0.4,k=5)

    if(len(most_relevant_file)>0):
        cl.user_session.set('response_msg',"Found the following topics relevant to your idea!")
        cl.user_session.set('options',list(set(most_relevant_file['name']))+['Create New','Select a Topic','Cancel'])
    else:
        cl.user_session.set('response_msg',"Found no topics relevant to your idea.")
        cl.user_session.set('options',['Create New','Select a Topic','Cancel'])

    async def choose_topic():

        response = await cl.AskActionMessage(
            content=cl.user_session.get('response_msg'),
            actions=[cl.Action(name=topic.lower(), 
                                payload={'name': topic.lower(), 'done':False, 'next':None, 'cancel':False}, 
                                label=uppercase(topic)) for topic in cl.user_session.get('options')],).send()
        payload = response['payload']

        # Response metadata
        if(payload['name']=='cancel'):
            payload['cancel'] = True
            return payload

        elif(payload['name']=='create new'):

            # Get the topic name as text from user
            topic_request = await cl.AskUserMessage(content="What is the topic of your idea?", timeout=30).send()
            topic_name = topic_request['output'].lower()

            # Check if the corresponding file already exists
            filename = re.sub(r'[^a-zA-Z0-9]', '_', topic_name)
            isExists = Path(f"{store_path}/{filename}.json").exists()
            if(isExists):
                # If the file exists, prompt the user for a different action
                payload['next'] = 'choose_topic'
                cl.user_session.set('response_msg','File already exists! Please select a different option.')
                cl.user_session.set('options',['Create New','Select a Topic','Cancel'])
                return payload

            payload['name'] = topic_name

        elif(payload['name']=='select a topic'):

            # Get the topic name as text from user
            topic_request = await cl.AskUserMessage(content="Choose a topic to update", timeout=30).send()
            topic_name = topic_request['output'].lower()

            # Check if the corresponding file already exists
            filename = re.sub(r'[^a-zA-Z0-9]', '_', topic_name)
            isExists = Path(f"{store_path}/{filename}.json").exists()
            if(not(isExists)):
                # If the file does not exist, prompt a different action
                payload['next'] = 'choose_topic'
                cl.user_session.set('response_msg','File does not exist! Please select a different option.')
                cl.user_session.set('options',['Create New','Select a Topic','Cancel'])
                return payload
            payload['name'] = topic_name

        print(payload['name'])
        payload['next'] = 'get_schema'
        return payload

    async def get_schema(payload):
        # Return a partially filled schema

        # If topic exists, return the saved schema, else, create a new one.
        topic_name = payload['name'].lower()
        filename = re.sub(r'[^a-zA-Z0-9]', '_', topic_name)
        isExists = Path(f"{store_path}/{filename}.json").exists()

        if(isExists):
            # Load existing schema
            with open(f"{store_path}/{filename}.json", "r") as file:
                data = json.load(file)
            schema = sch.get_schema(data['type'])
        else:
            # Create a new schema
            schemas = [uppercase(entity) for entity in named_entities]+['Cancel']

            response = await cl.AskActionMessage(
                content="What is kind of idea is it?",
                actions=[cl.Action(name=schema.lower(), payload={'name': schema.lower(), 'done':False, 'curr':'get_schema' ,'next':None, 'cancel':False}, label=schema) for schema in schemas],
            ).send()

            payload = response['payload']

            if(payload['name']=='cancel'):
                payload['cancel'] = True
                return payload

            # Retrieve the relevant schema
            schema_type = payload['name']
            schema = sch.get_schema(schema_type)

            data = dict()
            data['name'] = topic_name.lower()
            data['type'] = schema_type
            data['data'] = schema
            data['tags'] = []
            data['related'] = dict()
            data['aliases'] = []
            data['blurb'] = ''
            for key in data['data'].keys():
                data['related'][key] = []
            
        # Get the sub-schema associated with the idea
        sub_schemas = [uppercase(key) for key in schema.keys()]+['Cancel']

        response = await cl.AskActionMessage(
            content=f"What part of {uppercase(topic_name)} is your new idea about?",
            actions=[cl.Action(name=sub_schema.lower(), payload={'name': sub_schema.lower(), 'done':False, 'json':data, 'next':None, 'cancel':False}, label=sub_schema) for sub_schema in sub_schemas]
        ).send()
        payload = response['payload']

        if(payload['name']=='cancel'):
            payload['cancel'] = True
            return payload

        payload['next'] = 'done'
        payload['done'] = True

        return payload


    # Creation flow

    done = False
    next_step = 'choose_topic'
    while(not(done)):

        if(next_step=='choose_topic'):
            payload = await choose_topic()
        elif(next_step=='get_schema'):
            payload = await get_schema(payload)
            
        canceled = payload['cancel']
        if(canceled):
            break

        next_step = payload['next']
        done = payload['done']

    canceled = payload['cancel']
    print(canceled)

    if(not(canceled)):

        sub_schema = payload['name']

        data = payload['json']
        idea_name = data['name']
        idea_type = data['type']
        schema = data['data']

        props = {
                "timeout": 6000,
                "initialTab":sub_schema,
                "enableEdit": True,
                "initEdit": True,
                "topText": uppercase(idea_type),
                "Title": uppercase(idea_name),
                "fields": []}

        for key in schema.keys():
            new_field = dict()
            new_field['id'] = key
            new_field['label'] = uppercase(key)
            new_field['type'] = 'text'
            
            if(key.lower()==sub_schema.lower()):
                new_field['value'] = idea
            else:
                new_field['value'] = ''
            new_field['description'] = schema[key]#' '.join(schema[key])
            props['fields'].append(new_field)

        element = cl.CustomElement(
                        name="KnowledgeBase",
                        display="inline",
                        props=props
                    )

        response = await cl.AskElementMessage(
                    content="Please describe your idea!",
                    element=element,
                    timeout=6000
                ).send()

        # TODO : SAVE JSON to filesystem
        print(response)

        if(response['submitted']):
            for key in schema.keys():
                if(len(response[key])>0):
                    #schema[key].append(response[key])
                    schema[key] = response[key]

            print(schema)

            data['data'] = schema
            data['tags'] = []
            data['related'] = dict()

            for key in data['data'].keys():
                data['related'][key] = []

            filename = idea_name.lower()
            filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
            with open(f"{store_path}/json_store/{filename}.json", "w") as file:
                json.dump(data, file, indent=4)

            message = await cl.Message(content=f'Added idea : {idea}',actions=idea_actions).send()

            task_list = cl.user_session.get("task_list")

            idea_task = cl.Task(title=idea, status=cl.TaskStatus.READY)
            await task_list.add_task(idea_task)
            idea_task.forId = message.id
            print(len(task_list.tasks))

            # Update the task list in the interface
            await task_list.send()
    else:
        message = await cl.Message(content=f'Cancelled!').send()

    return None

async def update_metadata(user_input):
    actions = [
                cl.Action(
                    name="add_blurb",
                    icon="",
                    payload={'topic':user_input},
                    label="Add Blurb"
                ),
                cl.Action(
                    name="add_alias",
                    icon="",
                    payload={'topic':user_input},
                    label="Add Alias"
                ),
                cl.Action(
                    name="add_tag",
                    icon="",
                    payload={'topic':user_input},
                    label="Add Tag"
                ),
            ]

    exists,data = json_utils.load_json(user_input)
    if(exists):
        await cl.Message(content=f'Update metadata for {uppercase(user_input)}?',actions=actions).send()
    else:
        await cl.Message(content=f'No such file exists! You should create one.').send()

#############

# MAIN APP START

#############


@cl.on_settings_update
async def on_settings_update(settings: dict):

    cl.user_session.set("settings", settings)

    get_idea_history = settings['Get_Idea_History']

    if(get_idea_history):
        await show_saved_messages_panel()

@cl.on_chat_start
async def on_chat_start():

    await create_knowledge_graph()

    cl.user_session.set("alias_map",json_utils.get_alias_map())
    cl.user_session.set("blurb_map",json_utils.get_blurb_map())

    # Define System prompt
    cl.user_session.set("chat_history", [])
    cl.user_session.set("entities_to_track", [])

    await cl.context.emitter.set_commands(helpers)

    await cl.Message(content="Hello! I'm your novel-writing assistant. How can I help you today?",actions=persistent_actions).send()
    
    task_list = cl.TaskList()
    cl.user_session.set("task_list",task_list)
    task_list.status = "Idea History"
    task_list.title = "Ideas."

    await task_list.send()

@cl.on_message
async def on_message(user_message: cl.Message):


    selected_mode = user_message.command
    #user_message.modes.get("helper")
    """Main execution flow when user sends a message."""

    if(selected_mode=='Analyze'):
        if not user_message.elements:
            await cl.Message(content="No file attached").send()
        else:
            #async with cl.Step(name=f"Extract Entities") as step:
            #    await extract_entities(user_message)
            async with cl.Step(name=f"Adding Entity Context") as step:
                await get_gist(user_message)
                #await split_scenes(user_message)

    elif(selected_mode=='Update'):

        await update_faiss()

    elif(selected_mode=='Ideate'):
        await check_idea_for_contradictions(user_message.content)
        await save_idea_to_local_session(user_message.content)

    elif(selected_mode=='Check'):
        await check_idea(user_message.content)

    elif(selected_mode=='Metadata'):
        await update_metadata(user_message.content)

    else:

        user_input = user_message.content
        if(False):
            #user_input_with_context = await query_processing(user_input,entity_threshold=0.2)
            context_text,context_list = await retrieve_context(user_input)

            chat_history = []
            chat_history = prompts.get_decomposition_prompt('system',chat_history)
            chat_history = prompts.get_decomposition_prompt('user',chat_history,user_input,context_text)

            answer = await tokenize_and_generate(chat_history,max_new_tokens=256,temperature=0.4)

            print(answer)

        if(True):
            # Retrieve Base Context
            context_text,context_list = await retrieve_context(user_input)

            #sem.check_context_entailment(context_list,sub_queries)

            chat_history = []
            chat_history = prompts.get_generation_prompt('system',chat_history)
            chat_history = prompts.get_generation_prompt('user',chat_history,user_input,context_text)

            async with cl.Step(name=f"Local Context to Answer Question") as step:
                json_answer = await tokenize_and_generate(chat_history,max_new_tokens=1024,temperature=0.3,template=ReasonedResponse)

                response = ReasonedResponse.model_validate_json(json_answer)

                actions = [cl.Action(
                        name="show_reasoning",
                        icon="message-circle-question-mark",
                        payload={'content':response.scratchpad},
                        label="Show Reasoning"
                        )]
                await cl.Message(content=response.final_answer,actions=actions).send()

            # Check if Base Context is Sufficient
            best_context, max_prob = await check_context_sufficiency(response.scratchpad,context_list)

        
@cl.on_chat_end
def end_chat():

    chat_history = cl.user_session.get("task_list").tasks

    if(len(chat_history)>0):

        for key in chat_history.keys():

            idea = chat_history[key]

            file_path = '/Users/satyawagle/Projects/LangChain/llm_chatbot/data/obsidian_data/buffer_file.md'
            #file_path = action.payload['context']['path'][0]

            topic = 'LLM Brainstorming'

            lines_to_write = [f'- {idea}\n']

            # Check if the answer is already conveyed in the file (TODO).

            # Append to file
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            output_lines = []
            found_header = False
            appended = False

            target_header = "# LLM Brainstorming"
            for line in lines:
                output_lines.append(line)

                if(line.strip()==target_header):
                    found_header  = True

                # If we are under the target header and haven't appended the text yet
                if found_header and not appended:
                    # Check for the next block (either a new header or end of file)
                    if line.strip().startswith('#') and line.strip() != target_header:
                        # Insert text just before the next header
                        output_lines+=lines_to_write
                        appended = True
                    elif line == lines[-1]:
                        # If it's the last line, just append
                        output_lines+=lines_to_write
                        appended = True

            # If the text wasn't appended (e.g., EOF reached without next header), add it
            if found_header and not appended:
                output_lines+=lines_to_write

            if not found_header:
                output_lines.append('\n' + target_header + '\n')
                output_lines+=lines_to_write

            with open(file_path, 'w', encoding='utf-8') as f:
                f.writelines(output_lines)











