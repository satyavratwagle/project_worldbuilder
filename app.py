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
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,TorchAoConfig,AutoTokenizer,BartTokenizer, BartForConditionalGeneration
from pydantic import BaseModel, Field
from datasets import Dataset,concatenate_datasets
from typing import Literal
from sentence_transformers import SentenceTransformer
from utils.semantic import SemanticTools
from json_repair import repair_json
from pathlib import Path
from utils.io_utils import IO_Utils
from utils.pydantic_schema import ReasonedResponse,SceneSummary

from torchao.quantization import Int8WeightOnlyConfig, PerGroup
import mlx_lm
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import concurrent.futures
import gc
import mlx.core as mx
import psutil
from utils.datastore_utils import DatastoreUtilities

def uppercase(text):
    return text[0].upper()+text[1:]

def unload_models(model, tokenizer=None, embed_model=None):
    # 1. Delete the Python object references
    del model

    if(tokenizer):
        del tokenizer
    if embed_model is not None:
        del embed_model

    # 2. Force Python's garbage collector to clear reference cycles
    gc.collect()

    # 3. Clear PyTorch's Metal (MPS) cache (for SentenceTransformers / DeBERTa)
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    # 4. Clear MLX's Metal memory pool cache (for Llama models)
    mx.metal.clear_cache()
    
    print("Models successfully purged from memory!")

# Used by the response parser for tool
class RAGQuerySchema(BaseModel):
    query: str = Field(description="The search query to lookup local markdown files or notes.")

helpers = [{"id":"Ideate", "icon":"lightbulb", "description":"Add ideas to knowledge base"},
                {"id":"Analyze", "icon":"brain", "description":"Analyze uploaded text"},
                {"id":"Update", "icon":"list-restart", "description":"Update internal knowledge base"},
                {"id":"View", "icon":"eye", "description":"View a card"},
                {"id":"Metadata", "icon":"tag", "description":"Add Metadata"}]


with open('config.json', "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    config = json.load(f)
store_path= config['data_dir']


# Dedicated thread pool to satisfy MLX thread-local stream rules
mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

io_utils = IO_Utils()

model_id = "meta-llama/Llama-3.2-3B-Instruct"

named_entities = ['character','location','artifact','faction','event','definition']

#"Qwen/Qwen2.5-0.5B-Instruct" #
punctuation_tuple = tuple(string.punctuation)

# Engineering functions (synchronous)

@lru_cache(maxsize=32)
def get_or_create_generator(model, schema_class):
  return outlines.Generator(model,schema_class)

def _sync_generate(prompt, max_tokens, temperature, template):
    """The actual heavy MLX/Outlines code running safely in a background worker."""
    if template:
        generator = get_or_create_generator(outline_model, template)
        sampler = make_sampler(temp=temperature)
        return generator(prompt, sampler=sampler, max_tokens=max_tokens)
    else:
        sampler = make_sampler(temp=temperature)
        return generator_model(prompt, sampler=sampler, max_tokens=max_tokens)

def create_context(context_dict,topic):

    context_str = ''

    def key_description(key,desc):
        if(key in context_dict.keys()):
            return f'{desc} : {context_dict[key]}\n'
        else:
            return ''

    context_str += key_description('summary','Earlier scene')
    context_str += key_description('abilities',f'Estimated knowledge of the abilities possessed by {topic}')
    context_str += key_description('personality',f'Estimated knowledge of the personality of {topic}')
    context_str += key_description('backstory',f'Estimated knowledge of that past of {topic}')
    context_str += key_description('relationships',f'Estimated knowledge of the relationships of {topic} with other characters')

    return context_str


@cl.cache
def load_extraction_model():
    extraction_model_id = "urchade/gliner_small-v2.1"
    extraction_model = GLiNER.from_pretrained(extraction_model_id)

    return extraction_model

@cl.cache
def load_models():    

    #quant_config = Int8WeightOnlyConfig()
    #quantization_config = TorchAoConfig(quant_type=quant_config)
    #,quantization_config=quantization_config
    #hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps")
    #tokenizer = AutoTokenizer.from_pretrained(model_id)
    #model = outlines.from_transformers(hf_model, tokenizer)

    mlx_model, tokenizer = mlx_lm.load(config['model_dir']+'/'+config['model_id'])
    model = outlines.from_mlxlm(mlx_model, tokenizer)
    
    generator_model = outlines.Generator(model)
    #model = outlines.generate.regex(hf_modela, english_regex)

    return model, tokenizer, generator_model

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

outline_model, tokenizer, generator_model = load_models()
embed_model, dataset = load_embedding_models()
extraction_model  = load_extraction_model()
#nli_model,nli_tokenizer = load_semantic_consistency_models()

du = DatastoreUtilities(config)
du.load_embedding_model(embed_model,dataset)

sem = SemanticTools(config)
sem.load_extraction_model(extraction_model)

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

def create_json_dict(name,type):

    json_dict = dict()
    json_dict['name'] = name
    json_dict['type'] = type.lower()
    json_dict['data'] = sch.get_schema(type)
    json_dict['tags'] = []
    json_dict['related'] = sch.get_schema(type)
    json_dict['aliases'] = []
    json_dict['blurb'] = ""

    return json_dict

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

    return du.get_most_relevant_file(query,threshold,k)

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

    exists,data = du.load_json(action.payload['topic'])
    if(len(data['blurb'].strip())>0):
        await cl.Message(content=f'Currently : {data['blurb']}').send()
    res = await cl.AskUserMessage(content=f"Provide a description of {uppercase(action.payload['topic'])} in common words.",timeout=60).send()
    if(res):    
        if(exists):
            data['blurb'] = res['output']
            du.save_json(action.payload['topic'],data)
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

    exists,data = du.load_json(action.payload['topic'])
    if(len(data['aliases'])>0):
        await cl.Message(content=f'Currently : {','.join(data['aliases'])}').send()
    res = await cl.AskUserMessage(content=f"Provide an alias for {uppercase(action.payload['topic'])}.",timeout=60).send()

    if(res):
        if(exists):
            aliases = res['output'].split(',')
            data['aliases'] += aliases
            data['aliases'] = list(set(data['aliases']))
            du.save_json(action.payload['topic'],data)
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

    exists,data = du.load_json(action.payload['topic'])
    if(len(data['tags'])>0):
        await cl.Message(content=f'Currently : {','.join(data['tags'])}').send()
    res = await cl.AskUserMessage(content=f"Provide tags for {uppercase(action.payload['topic'])}.",timeout=60).send()

    if(res):
        if(exists):
            aliases = res['output'].split(',')
            data['tags'] += aliases
            du.save_json(action.payload['topic'],data)
            await cl.Message(content=f'Saved tags for {uppercase(action.payload['topic'])}',actions=actions).send()
        else:
            await cl.Message(content=f'File not found!').send()

    await cl.context.emitter.task_end()
        
async def show_checklist(entities,message='Select topics.'):

    # Entities : Dict: entity:label (e.g "Alvar" : "character")

    items_list = []
    for key in entities.keys():

        entity = key
        label = entities[key]

        items_list.append(
                {
                    "id":entity,
                    "label":entity,
                    "description":label,
                    "defaultChecked": False
                })

    props = {
            "timeout": 6000,
            "topText": "Found the following topics in the text!",
            "Title": "Select topics to track!",
            "items": items_list}

    checklist_element = cl.CustomElement(
        name="SelectToTrack",
        props=props
    )

    element_msg = cl.AskElementMessage(
        content=message,
        element=checklist_element
    )
    # 3. Send the component attached to a chat message
    selection_response = await element_msg.send()

    chosen_selections = []
    for key in selection_response.keys():
        if(not(key=='submitted') and selection_response[key]):
            chosen_selections.append(key)

    element_msg.content = f'Selected {', '.join(chosen_selections)}!'
    await element_msg.update()

    return selection_response

async def show_card(json_dict,message='Please describe your idea!',open_with='overview',initEdit=False,enableEdit=True):

    # card_title - Name ("Alvar")
    # crd_type - Idea type ("Character")
    # open_with - topic to open on ("appearance")
    # schema - dict() containing all sub_schema
    # idea (optional) - dict() like schema : Stuff to prefill


    # Check if file exists. If it does not, add an empty schema.
    filename = re.sub(r'[^a-zA-Z0-9]', '_', json_dict['name'].lower())
    isExists = Path(f"{store_path}/{filename}.json").exists()
    if(isExists):
        # Load existing schema
        with open(f"{store_path}/{filename}.json", "r") as file:
            data = json.load(file)
        prefill = data['data']

    else:
        prefill = sch.get_schema(json_dict['type'])

    props = {
            "timeout": 6000,
            "initialTab":open_with,
            "enableEdit": enableEdit,
            "initEdit": initEdit,
            "topText": uppercase(json_dict['type']),
            "Title": uppercase(json_dict['name']),
            "fields": []}

    for key in json_dict['data'].keys():
        new_field = dict()
        new_field['id'] = key
        new_field['label'] = uppercase(key)
        new_field['type'] = 'text'
        new_field['value'] = prefill[key]#json_dict['data'][key]
        new_field['description'] = json_dict['data'][key]#prefill[key]
        props['fields'].append(new_field)

    element = cl.CustomElement(
                    name="KnowledgeBase",
                    display="inline",
                    props=props
                )

    card_element = cl.AskElementMessage(
                content=message,
                element=element,
                timeout=6000
            )

    response = await card_element.send()

    card_element.content = f'Saved new information about {uppercase(json_dict['name'])}!'
    await card_element.update()

    return response

async def save_json_to_database(json_dict):
    # Given a valid json_dict of information, saves it to database.
    # json_dict : dict() 'name':str,'type':str,'data':dict,'related':dict...

    filename = json_dict['name'].lower()
    filename = re.sub(r'[^a-zA-Z0-9]', '_', filename)
    isExists = Path(f"{store_path}/{filename}.json").exists()
    if(isExists):
        with open(f"{store_path}/{filename}.json", "r") as file:
            loaded_file = json.load(file)
    else:
        loaded_file = create_json_dict(json_dict['name'],json_dict['type'])
        loaded_file['name'] = json_dict['name'].lower()

    for key in loaded_file['data']:
        loaded_file['data'][key].append(', '.join(json_dict['data'][key]))

    with open(f"{store_path}/json_store/{filename}.json", "w") as file:
        json.dump(loaded_file, file, indent=4)

#############
# Datastore Operations
#############

@cl.step(name="Knowledge Database Update Tools")
async def update_datastore():

    await create_knowledge_graph()
    await cl.make_async(du.update_local_dataset)()    

async def create_knowledge_graph():
    graph,documents_lookup,filename_to_name = du.create_knowledge_graph()
    cl.user_session.set('knowledge_graph',graph)
    cl.user_session.set('documents_lookup',documents_lookup)

async def save_idea_to_local_session(message):    

    idea = message.content

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

        response = await show_card(payload['json'],message='Please describe your idea!',open_with=payload['name'],initEdit=False,enableEdit=True)

        '''sub_schema = payload['name']

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
                ).send()'''

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
    else:
        message = await cl.Message(content=f'Cancelled!').send()

    return None

async def update_metadata(message):
    actions = [
                cl.Action(
                    name="add_blurb",
                    icon="",
                    payload={'topic':message.content},
                    label="Add Blurb"
                ),
                cl.Action(
                    name="add_alias",
                    icon="",
                    payload={'topic':message.content},
                    label="Add Alias"
                ),
                cl.Action(
                    name="add_tag",
                    icon="",
                    payload={'topic':message.content},
                    label="Add Tag"
                ),
            ]

    exists,data = du.load_json(message.content)
    if(exists):
        await cl.Message(content=f'Update metadata for {uppercase(message.content)}?',actions=actions).send()
    else:
        await cl.Message(content=f'No such file exists! You should create one.').send()

async def retrieve_graph_rag(query,threshold=0.4,k=10,hops=1):

    graph = cl.user_session.get("knowledge_graph")
    documents_lookup = cl.user_session.get("documents_lookup")

    retrieved_ids = set()
    full_graph_context = dict()
    full_graph_context['direct'] = []
    full_graph_context['indirect'] = []

    
    retrieved_contexts,ids = du.get_graph_rag_context(query,graph,documents_lookup,threshold,k,hops)

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
        full_graph_context['direct'].reverse()
        full_context_text += f"DIRECT CONTEXT:\n\n{"\n\n".join(full_graph_context['direct'])}"

    if(len(full_context_text)==0):
        full_context_text = "No context found."

    return  full_context_text,full_graph_context

@cl.step(name='Retrieve Context')
async def retrieve_context(topic):

    #chat_history = []
    #chat_history = prompts.get_context_retrieval_prompt('system',chat_history)
    #chat_history = prompts.get_context_retrieval_prompt('user',chat_history,user_input,', '.join(relevant_entities))

    # Construct Tool Call
    relevant_entities = sem.entity_extraction(topic,entity_threshold=0.25)

    # Context Retrievant Via MCP
    topic_with_context = await query_processing(topic)
    context_text, context_list = await retrieve_graph_rag(topic_with_context)

    return context_text,context_list

async def view_idea(message):

    exists,data = du.load_json(message.content)

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

#############
# Corpus Analysis Tools
#############

@cl.step(name='Splitting Scenes')
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

    chunk_embeddings = du.embed_text(chunks)

    for idx in range(len(chunk_embeddings)-1):
        cosine_sim = du.get_cosine_similarity(chunk_embeddings[idx],chunk_embeddings[idx+1])
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
    entities = sem.named_entity_extraction(io_utils.get_text(file.path))
    selection_response = await show_checklist(entities,message='Please select the entities to learn about!')
    
    entities_to_track = dict()
    for key in selection_response.keys():
        if(not(key=='submitted') and selection_response[key]):
            entities_to_track[key] = entities[key] # Harry : character

    aliases = dict()
    for key in entities_to_track:
        aliases[key] = sem.find_aliases(key,io_utils.get_text(file.path))
        if(len(aliases[key])>0):
            await cl.Message(content=f'Found the following aliases for {key}!\n {'\n'.join([f"{i}. {uppercase(alias)}" for i, alias in enumerate(aliases[key], start=1)])}').send()
        else:
            alias_text = await cl.AskUserMessage(content=f'Found no aliases for {key}! Would you like to add any aliases? Please split each alias with a /.',timeout=120).send()
            aliases[key] = [key]

            if('/' in alias_text['output']):
                aliases[key] += alias_text['output'].split('/')
            else:
                aliases[key] += [alias_text['output']]

    print(aliases)


    scenes = pre.get_chunks(pre.get_corpus(file.path),chunk_size=800)#await split_scenes(file)

    tracked_entities = dict()
    entity_summaries = dict()

    for key in entities_to_track.keys():
        type = entities_to_track[key]
        tracked_entities[key] = dict()

        previous_context = None

        for scene_number in range(len(scenes)):
            scene = scenes[scene_number]

            with cl.Step(f'Analysis Tools on Scene : {scene[:min((50,len(scene)))]}...'):
                #with cl.Step(f'Analysis of Scene : {scene}'):
                
                tracked_entities[key][scene_number] = create_json_dict(key,type)

                if(any(alias in scene for alias in aliases[key])): 
                    get_full = True
                else: 
                    get_full = False
                    for sch_key in tracked_entities[key][scene_number]['data'].keys():
                        tracked_entities[key][scene_number]['data'][sch_key] = ''


                RoleSchema = sch.get_pydantic_schema('RoleSchema',type,key,full=get_full)
                subjects = list(RoleSchema.model_fields.keys())

                entity_extraction_prompt_history = prompts.isolate_scene_element(role='system')
                entity_extraction_prompt_history = prompts.isolate_scene_element(role='user',history=entity_extraction_prompt_history,text=scene,entity=key,subjects=subjects,aliases=aliases[key],context=previous_context,full=get_full)
                print(entity_extraction_prompt_history)
                isolated_element = await tokenize_and_generate(entity_extraction_prompt_history,max_new_tokens=256,temperature=0.25,template=RoleSchema)
                isolated_element = RoleSchema.model_validate_json(isolated_element)
                isolated_element = isolated_element.model_dump()
                previous_context = create_context(isolated_element,key)
                for sch_key in isolated_element.keys():
                    print(sch_key,isolated_element[sch_key])
                print()

                for sch_key in isolated_element.keys():
                    tracked_entities[key][scene_number]['data'][sch_key] = isolated_element[sch_key]
        

        with cl.Step(f'Summarization Tools on information about {key}'):
            entity_summaries[key] = create_json_dict(key,type)

            for sch_key in entity_summaries[key]['data'].keys():
                if(not(sch_key=='misc')):
                    print([tracked_entities[key][s]['data'][sch_key] for s in tracked_entities[key].keys()])
                    joint_text = ', '.join([tracked_entities[key][s]['data'][sch_key] for s in tracked_entities[key].keys()])
                    summarization_prompt_history = prompts.get_summarization_prompt(role='system')
                    summarization_prompt_history = prompts.get_summarization_prompt(role='user',history=summarization_prompt_history,entity=key,type=type,subject=sch_key,text=joint_text)
                    entity_summary = await tokenize_and_generate(summarization_prompt_history,max_new_tokens=256,temperature=0.5)
                    print(sch_key,entity_summary)
                    entity_summaries[key]['data'][sch_key] = [entity_summary]


            response = await show_card(entity_summaries[key],message=f'I was able to find the following information about {key}!')

            if(response['submitted']):
                for sch_key in entity_summaries[key]['data']:
                    entity_summaries[key][sch_key] = response[sch_key]
                await save_json_to_database(entity_summaries[key])
            else:
                await cl.Message(content='Cancelled Wiki Creation Process!').send()


    '''
    

    summary = 'No context available.' 
    tracked_entities = dict()
    scene_summaries = dict()

    scene_number = 1
    for scene in scenes:
        #scene_with_context = await add_entity_context(scene)
        gist_prompt_history = prompts.get_gist_prompt(role='system') 
        gist_prompt_history = prompts.get_gist_prompt(role='user',history=gist_prompt_history,text=scene,tracked_entities=list(tracked_entities.keys()))
        summary = await tokenize_and_generate(gist_prompt_history,max_new_tokens=256,temperature=0.4,template=SceneSummary)
        scene_summary = SceneSummary.model_validate_json(summary)
        scene_summary = scene_summary.model_dump()

        print(scene_summary)

        found_subjects = dict()
        for key in scene_summary:
            if(not(key=='summary')):
                for subject in scene_summary[key]:
                    found_subjects[subject] = key

        #scene_summaries[scene_number] = scene_summary['summary']

        selection_response = await show_checklist(found_subjects,message='In Scene')
        print(selection_response)

        if(selection_response['submitted']):
            for key in selection_response:
                if(selection_response[key] and not(key=='submitted') and not(key in tracked_entities.keys())):
                    tracked_entities[key] = dict()
                    tracked_entities[key][scene_number] = create_json_dict(key,found_subjects[key])
                elif(key in tracked_entities.keys()):
                    tracked_entities[key][scene_number] = create_json_dict(key,found_subjects[key])

        else:
            pass

        print(tracked_entities)

        # Redesign the summary to feed back as context.
        for entity in tracked_entities.keys():

            tracked_scene = list(tracked_entities[entity].keys())[-1]

            if(tracked_scene==scene_number):
                role = tracked_entities[entity][tracked_scene]['type']

                RoleSchema = sch.get_pydantic_schema('RoleSchema',role,entity)
                role_subjects = list(RoleSchema.model_fields.keys())

                scene_extraction_prompt_history = prompts.isolate_scene_element(role='system')
                scene_extraction_prompt_history = prompts.isolate_scene_element(role='user',history=scene_extraction_prompt_history,text=scene,entity=entity,subjects=role_subjects)
                isolated_element = await tokenize_and_generate(scene_extraction_prompt_history,max_new_tokens=256,temperature=0.25,template=RoleSchema)
                print(isolated_element)
                isolated_element = RoleSchema.model_validate_json(isolated_element)
                isolated_element = isolated_element.model_dump()

                for key in tracked_entities.keys():
                    for sch_key in isolated_element.keys():
                        tracked_entities[key][tracked_scene]['data'][sch_key] = isolated_element[sch_key]

            print(tracked_entities)
        scene_number += 1

    # Consolidate information in tracked entities - Use an LLM to do this later.

    entity_summaries = dict()
    for key in tracked_entities.keys():
        scene = list(tracked_entities[key].keys())[0]
        role = tracked_entities[key][scene]['type']
        entity = tracked_entities[key][scene]['name']
        entity_summaries[key] = create_json_dict(key,tracked_entities[key][scene]['type'])
        for sch_key in entity_summaries[key]['data'].keys():
            if(not(sch_key=='overview') and not(sch_key=='misc')):
                print(sch_key)
                entity_summaries[key]['data'][sch_key] = ', '.join([tracked_entities[key][s]['data'][sch_key] for s in tracked_entities[key].keys()])

                summarization_prompt_history = prompts.get_summarization_prompt(role='system')
                summarization_prompt_history = prompts.get_summarization_prompt(role='user',history=summarization_prompt_history,entity=entity,type=role,text='; '.join([tracked_entities[key][s]['data'][sch_key] for s in tracked_entities[key].keys()]))
                entity_summary = await tokenize_and_generate(summarization_prompt_history,max_new_tokens=256,temperature=0.5)
                print(entity_summary)



    #print(entity_summaries)
    '''

async def extract_entities(user_message):

    tracked_entities = cl.user_session.get("entities_to_track")
    file = user_message.elements[0]

    selection_response = await show_checklist(entities)
    print(selection_response)

    for key in selection_response.keys():
        if(not(key.lower()=='submitted') and selection_response[key]):

            entity = key.lower()
            label = entities[key]

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


#############
# Chainlit message manipulation tools (?)
#############

async def delete_last_message():
    chat_context = cl.chat_context.get()
    await chat_context[-1].remove()


#############
# Semantic Functions
#############

async def add_entity_context(idea,entity_threshold=0.4):

    blurb_map = cl.user_session.get('blurb_map')
    alias_map = cl.user_session.get('alias_map')

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

#############
# Core Functions
#############

async def tokenize_and_generate(chat_history,max_new_tokens=256,temperature=0.6,template=None, use_chat_template=True):

    synthesis_prompt = tokenizer.apply_chat_template(
                                chat_history,
                                tokenize=False,
                                add_generation_prompt=True
                                )

    loop = asyncio.get_running_loop()
    answer = await loop.run_in_executor(
        mlx_executor, 
        _sync_generate, 
        synthesis_prompt, 
        max_new_tokens, 
        temperature, 
        template
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

async def check_idea_for_contradictions(message):

    idea = message.content

    _,context_dict = await retrieve_context(idea)

    _,max_prob = await check_context_sufficiency(idea,context_dict)

@cl.step(name='Local Context to Reason and Answer')
async def reason_and_answer(message):

    # Retrieve Base Context
    context_text,context_list = await retrieve_context(message.content)

    chat_history = []
    chat_history = prompts.get_generation_prompt('system',chat_history)
    chat_history = prompts.get_generation_prompt('user',chat_history,message.content,context_text)

    json_answer = await tokenize_and_generate(chat_history,max_new_tokens=1024,temperature=0.3,template=ReasonedResponse)

    response = ReasonedResponse.model_validate_json(json_answer)

    actions = [cl.Action(
            name="show_reasoning",
            icon="message-circle-question-mark",
            payload={'content':response.scratchpad},
            label="Show Reasoning"
            )]
    await cl.Message(content=response.final_answer,actions=actions).send()

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
    await cl.make_async(du.reload_faiss_dataset)()

    cl.user_session.set("alias_map",du.get_alias_map())
    cl.user_session.set("blurb_map",du.get_blurb_map())

    # Define System prompt
    cl.user_session.set("chat_history", [])
    cl.user_session.set("entities_to_track", [])

    await cl.context.emitter.set_commands(helpers)

    await cl.Message(content="Hello! I'm Quill, your novel-writing assistant! How can I help you today?").send()

@cl.on_message
async def on_message(user_message: cl.Message):


    selected_mode = user_message.command
    #user_message.modes.get("helper")
    """Main execution flow when user sends a message."""

    if(selected_mode=='Analyze'):
        if not user_message.elements:
            await cl.Message(content="No file attached").send()
        else:
            await get_gist(user_message)
    elif(selected_mode=='Update'):
        await update_datastore()
    elif(selected_mode=='Ideate'):
        #await check_idea_for_contradictions(user_message)
        await save_idea_to_local_session(user_message)
    elif(selected_mode=='View'):
        await view_idea(user_message)
    elif(selected_mode=='Metadata'):
        await update_metadata(user_message)
    else:
        await reason_and_answer(user_message)
     
@cl.on_chat_end
async def end_chat():
    await update_datastore()