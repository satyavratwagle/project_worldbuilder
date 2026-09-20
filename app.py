import chainlit as cl
from chainlit.input_widget import Slider
import asyncio
import json
import torch
import numpy as np

from functools import lru_cache
import re
import sys
import os
import datasets
import outlines
import logging
import frontmatter
import shutil
import networkx as nx
import matplotlib.pyplot as plt
import ahocorasick
import string
import ast

import utils.prompts as prompts
from utils.semantic import SemanticTools
import utils.json_schema as sch
import utils.preprocessing as pre
from utils.io_utils import IO_Utils
from utils.pydantic_schema import ReasonedResponse,HypothesisList
from utils.datastore_utils import DatastoreUtilities

from gliner import GLiNER
from transformers import AutoModelForCausalLM,AutoModelForSequenceClassification,TorchAoConfig,AutoTokenizer,BartTokenizer, BartForConditionalGeneration
from transformers import AutoConfig
from pydantic import BaseModel, Field
from datasets import Dataset,concatenate_datasets
from typing import Literal
from sentence_transformers import SentenceTransformer
from json_repair import repair_json
from pathlib import Path

from torchao.quantization import Int8WeightOnlyConfig, PerGroup
import mlx_lm
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler
import concurrent.futures
import gc
import mlx.core as mx
import psutil
import difflib
import time

helpers = [ {"id":"Ideate", "icon":"lightbulb", "description":"Add ideas to knowledge base"},
            {"id":"Brainstorm", "icon":"lightbulb", "description":"Brainstorm for new ideas"},
            {"id":"Analyze", "icon":"brain", "description":"Analyze uploaded text"},
            {"id":"Update", "icon":"list-restart", "description":"Update internal knowledge base"},
            {"id":"View", "icon":"eye", "description":"View a card"},
            {"id":"Metadata", "icon":"tag", "description":"Add Metadata"},
            {"id":"Forget", "icon":"trash", "description":"Forget the Current Conversation"}]

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
    wiki_schema = json.load(f)

outline_model = None
tokenizer = None
generator_model = None

mlx_executor = None

io_utils = IO_Utils()
named_entities = ['character','location','artifact','faction','event','definition']

#"Qwen/Qwen2.5-0.5B-Instruct" #
punctuation_tuple = tuple(string.punctuation)

# Engineering functions (synchronous)

@lru_cache(maxsize=32)
def get_or_create_generator(model, schema_class):
  return outlines.Generator(model,schema_class)

def _sync_load_models():
    """Loads models entirely inside the background worker thread so the stream belongs to it."""
    global outline_model, tokenizer, generator_model
    if outline_model is None:
        mlx_model, tokenizer_obj = mlx_lm.load(config['model_dir'] + '/' + config['model_id'])
        model_obj = outlines.from_mlxlm(mlx_model, tokenizer_obj)
        generator_model_obj = outlines.Generator(model_obj)
        
        outline_model = model_obj
        tokenizer = tokenizer_obj
        generator_model = generator_model_obj
    return outline_model, tokenizer, generator_model

def _sync_generate(prompt, max_tokens, temperature, template):
    """The actual heavy MLX/Outlines code running safely in a background worker."""
    _sync_load_models()
    mx.eval()
    outlines.caching.clear_cache()

    with torch.no_grad():
        if template:
            generator = get_or_create_generator(outline_model, template)
            sampler = make_sampler(temp=temperature)
            return generator(prompt, sampler=sampler, max_tokens=max_tokens)
        else:
            sampler = make_sampler(temp=temperature)
            return generator_model(prompt, sampler=sampler, max_tokens=max_tokens)

def create_json_dict(name,type):

    json_dict = dict()
    json_dict['name'] = name
    json_dict['type'] = type.lower()
    json_dict['data'] = sch.get_schema(type.lower())
    json_dict['tags'] = []
    json_dict['related'] = sch.get_schema(type.lower())
    json_dict['aliases'] = []
    json_dict['blurb'] = ""

    return json_dict

def uppercase(text):
    return text[0].upper()+text[1:]

async def update_graph_element(node_info):

    print('Yeah')
    graph_element = cl.user_session.get("graph_element")
    graph_element.props['title'] = node_info['node_name']
    graph_element.props['subtitle'] = node_info['node_type']
    graph_element.props['edges'] = node_info['edge_info']
    graph_element.props['wiki'] = node_info['wiki']
    print(graph_element.props)
    cl.user_session.set("graph_element", graph_element)
    await graph_element.update()

    return graph_element

@cl.cache
def load_extraction_model():
    #extraction_model_id = config["gliner_dir"]+'/checkpoint-1000'
    extraction_model_id = "knowledgator/gliner-relex-large-v1.0"
    extraction_model = GLiNER.from_pretrained(extraction_model_id)
    #extraction_model.config.max_span_width = 

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
def load_nli_models(): 

    nli_model_id        = "cross-encoder/nli-deberta-v3-large"
    nli_max_length = 512

    # Load and update the configuration to accommodate larger token lengths
    config = AutoConfig.from_pretrained(nli_model_id)
    config.max_position_embeddings = nli_max_length
    config.max_relative_positions = nli_max_length
    
    nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
    nli_tokenizer.model_max_length = nli_max_length
    nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id,config=config)

    return nli_model,nli_tokenizer

#outline_model, tokenizer, generator_model = load_models()
#embed_model = load_embedding_models()
extraction_model  = load_extraction_model()

du = DatastoreUtilities(config)
du.load_embedding_model()

sem = SemanticTools(config)
sem.load_extraction_model(extraction_model)

nli_model,nli_tokenizer = load_nli_models()
sem.load_nli_model(nli_model,nli_tokenizer)
sem.load_zsc_model("knowledgator/gliclass-modern-base-v3.0")

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

# Legacy
async def get_most_relevant_file(query: str, threshold: float = 0.4, k = 1) -> str:

    return du.get_most_relevant_file(query,threshold,k)

#############

# Chainlit UI Functions

#############

async def delete_last_message():
    chat_context = cl.chat_context.get()
    await chat_context[-1].remove()

async def show_checklist(entities,message='Select topics.',show_description=True,show_response=True):

    # Entities : Dict: entity:label (e.g "Alvar" : "character")

    items_list = []
    for key in entities.keys():

        entity = key
        label = entities[key]

        if(show_description):
            items_list.append(
                    {
                        "id":entity,
                        "label":entity,
                        "description":label,
                        "defaultChecked": False
                    })
        else:
            items_list.append(
                    {
                        "id":label,
                        "label":entity,
                        "description":"",
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
        element=checklist_element,
        timeout=360
    )
    # 3. Send the component attached to a chat message
    selection_response = await element_msg.send()

    chosen_selections = dict()
    for key in selection_response.keys():
        if(not(key=='submitted')):
            if(selection_response[key][2]):
                chosen_selections[key] = selection_response[key][1]

    if(show_response):
        element_msg.content = f'Selected {', '.join(chosen_selections.keys())}!'
        await element_msg.update()

    return chosen_selections

async def show_edges_to_add(sentences,message='Select topics.',show_description=True):
    # Entities : Dict: entity:label (e.g "Alvar" : "character")

    items_list = []
    for idx,s in enumerate(sentences):
        if(len(s.strip())>0):
            items_list.append({"id":idx,"text":s})

    
    props = {
            "timeout": 6000,
            "topText": "Select Info to Add!",
            "Title": "Select Info to Add!!",
            "items": items_list}

    checklist_element = cl.CustomElement(
        name="AddToGraph",
        props=props
    )

    element_msg = cl.AskElementMessage(
        content=message,
        element=checklist_element
    )
    # 3. Send the component attached to a chat message
    selection_response = await element_msg.send()

    edges_to_add = []
    for key in selection_response.keys():
        if(not(key=='submitted')):
            if(selection_response[key][1]):
                edges_to_add.append(selection_response[key][0])

    return edges_to_add

async def show_summaries_to_add(summaries,message='Select topics.',show_description=True):
    # Entities : Dict: entity:label (e.g "Alvar" : "character")

    items_list = []
    for idx,entry in enumerate(summaries):
        node,node_name,summary,wiki_key = entry
        if(len(summary.strip())>0):
            items_list.append({"id":idx,"node":node,"node_name":node_name,"text":summary.strip(),"wiki_key":wiki_key})

    
    props = {
            "timeout": 6000,
            "topText": "Select Info to Add!",
            "Title": "Select Info to Add!!",
            "items": items_list}

    checklist_element = cl.CustomElement(
        name="SummarySelectionElement",
        props=props
    )

    element_msg = cl.AskElementMessage(
        content=message,
        element=checklist_element
    )
    # 3. Send the component attached to a chat message
    selection_response = await element_msg.send()

    summaries_to_add = []
    for key in selection_response.keys():
        if(not(key=='submitted')):
            if(selection_response[key][1]):

                # Add node name and key here
                summaries_to_add.append((selection_response[key][0],selection_response[key][5],selection_response[key][1]))
                print(selection_response[key])


    return summaries_to_add

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

@cl.action_callback("switch_node")
async def on_edge_click(action: cl.Action):
    payload = action.payload
    node_name = payload.get("node")

    print(node_name)
    node_info = du.get_node_info(node_name)
    print(node_info)    
    if(node_info):
        graph_element = await update_graph_element(node_info)

        #cl.user_session.set("graph_message", graph_message)
        #response = await graph_message.send()

@cl.action_callback("edit_edge")
async def on_update_edge(action: cl.Action):
    payload = action.payload
    head = payload.get("head")
    tail = payload.get("tail")
    key = payload.get("key")
    text = payload.get("text")

    cl.user_session.set("awaiting_input_edge", True)
    cl.user_session.set("payload", payload)
    response = await cl.Message(content=f"Please modify '{text}'' in the chat.").send()

@cl.action_callback("update_node_summary")
async def on_update_node(action: cl.Action):

    print('Here!')
    payload = action.payload
    node = payload.get("node")
    summary = payload.get("summary")
    
    cl.user_session.set("awaiting_input_node", True)
    cl.user_session.set("payload", payload)
    response = await cl.Message(content=f"Please provide a new summary for {node} in the chat.\n Current Summary is : '{summary}'").send()

@cl.action_callback("exit")
async def on_exit(action: cl.Action):
    payload = action.payload
    print(payload)
    save = payload.get("save")
    if(save):
        du.save_graph()
        du.reload_graph()
    else:
        du.reload_graph()

    graph_message = cl.user_session.get("graph_message")
    await graph_message.remove()

@cl.action_callback("delete_edge")
async def on_delete_edge(action: cl.Action):
    payload = action.payload
    head = payload.get("head")
    tail = payload.get("tail")
    key = payload.get("key")
    text = payload.get("text")

    print(du.knowledge_graph)
    du.knowledge_graph.remove_edges_from([(head,tail,key)])
    print(du.knowledge_graph)

    node_info = du.get_node_info(head)

    if(node_info):
        graph_element = await update_graph_element(node_info)

    await cl.Message(f"Deleted '{text}' !").send()

@cl.action_callback("delete_node")
async def on_delete_node(action: cl.Action):
    payload = action.payload
    node = payload.get("node")
    
    print(du.knowledge_graph)
    du.remove_node(node)
    print(du.knowledge_graph)

    node_info = du.get_node_info(du.get_all_nodes()[0])

    if(node_info):
        graph_element = await update_graph_element(node_info)

    await cl.Message(f"Deleted '{node}' !").send()
    
@cl.action_callback("show_context")
async def on_show_context(action: cl.Action):
    context_edges = action.payload.get("context_edges")

    graph_element = cl.user_session.get("graph_element")
    graph_element.props['title'] = "Context Used in Answer"
    graph_element.props['subtitle'] = ""
    graph_element.props['edges'] = context_edges

    graph_message = cl.Message(
                content="Showing node information!",
                elements=[graph_element]
            )

    cl.user_session.set("graph_message", graph_message)

    response = await graph_message.send()

@cl.action_callback("show_node_info")
async def on_show_node_info(action: cl.Action):

    node_name = action.payload.get("node_name")
    node_info = du.get_node_info(node_name)

    if(node_info):

        graph_element = await update_graph_element(node_info)

        graph_message = cl.Message(
                    content="Showing node information!",
                    elements=[graph_element]
                )

        cl.user_session.set("graph_message", graph_message)

        response = await graph_message.send()

    else:
        await cl.Message(content=f'No node named {node_name} found!').send()
#############

# Datastore Operations

#############

@cl.action_callback("update_faiss")
async def update_datastore(action: cl.Action):

    async with cl.Step(name="Saving Knowledge Base",default_open=True) as step:
        dataset,index = du.update_faiss_dataset()    
        du.overwrite_faiss_dataset(dataset,index)
        du.reload_faiss_dataset()

async def retrieve_graph_rag(query,threshold=0.4,k=10,hops=1):

    retrieved_context_list,context_edges = du.get_graph_rag_context(query,threshold,k)

    return '\n'.join(retrieved_context_list),context_edges

#############

# Knowledge Graph Tools

#############

async def extract_knowledge_graph(og_text,graph_name='graph',graph_type='story'):

    if(graph_type=='idea'):
        graph_name = 'lore_graph'

    async with cl.Step(name="Topic Selection",default_open=True) as step:
        extracted_pos,doc = sem.extract_pos(og_text)
        head,relation,tail = sem.get_triplets(extracted_pos,doc)
        text,clusters,doc = sem.get_coref_clusters(og_text)

        '''aliased_entities = []
        for cluster in clusters:

            temp_cluster = cluster+['Choose New','Skip']
            response = await cl.AskActionMessage(
                content = 'What alias should be used?',
                actions = [cl.Action(name=name,payload={'alias':name},label=name) for name in temp_cluster],
                timeout = 60).send()

            if(response['name']=='Choose New'):
                response = await cl.AskUserMessage(
                content = 'Please provide a new Alias!',
                timeout = 60).send()
                aliased_entities.append((cluster,response['output']))
            elif(not(response['name']=='Skip')):
                aliased_entities.append((cluster,response['name']))

        #await cl.Message(content=og_text).send()
        resolved_text = sem.resolve_coreferences(og_text,aliased_entities)
        await cl.Message(content=resolved_text).send()'''

        resolved_text = og_text

        # Extract (Named Entities from Text + Nodes from Graph + Additional User Specified Entities)
        extracted_entities = sem.named_entity_extraction(resolved_text,labels=['Character','Location','Artifact','Faction','Event'])
        similar_pairs = du.check_nodes_for_replacement(extracted_entities.keys())        

        # Get user response on what to replace
        replacements = []
        for key,node in similar_pairs:
            actions = [cl.Action(name=name,payload={'alias':name},label=name) for name in ['Yes','No']]
            response = await cl.AskActionMessage(content=f"'{key}' is similar to existing node '{node['name']}'. Replace?",actions=actions,timeout=30).send()
            if(response['name']=='Yes'):
                replacements.append((key,node['name']))

            for old_key,new_key in replacements:
                if(old_key in extracted_entities.keys()):
                    extracted_entities[new_key] = extracted_entities.pop(old_key)


        # Get user to add More Entities to the nodes to extract
        selection_response = await show_checklist(extracted_entities,message='Please select the topics to add to the story graph!')
        extracted_entity_names = dict()
        user_defined_entity_names = dict()
        for k,v in selection_response.items():
            print(k,v)
            extracted_entity_names[k] = v
            
        
    async with cl.Step(name="Extracting Information from Scene",default_open=True) as step:
        # Extract node names from the existing Graph

        # Combine all names
        topic_dict = extracted_entity_names | user_defined_entity_names
        print(topic_dict)
        
        alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias,type in topic_dict.items())})\b"
        alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

        # Summarize the scenes
        chunk_idx = 1
        for chunk in resolved_text.split('\n\n'):

            entities = alias_finder.findall(chunk)

            # Change the type of the entity if needed.
            #scene_entities = [e['text'] for e in entities[0] if e['text'] in aliased_entity_names]+user_defined_entity_names
            resolved_text = await decompose_text(chunk,topics=list(set(entities)))
            
            # Resolve coreferences in LLM output if any
            #resolved_text,clusters,_ = sem.get_coref_clusters(og_text+'\n\n'+text)
            #resolved_text = sem.resolve_coreferences(og_text+'\n\n'+resolved_text,aliased_entities).split('\n\n')[-1]

            # Check for redundant sentences
            split_sentences = resolved_text.split('.')[:-1]
            cosine_sim = du.get_cosine_similarity(split_sentences)
            filtered_sentences,_ = du.filter_similar_text(split_sentences,cosine_sim)

            #await cl.Message(content=f"Summary\n- {'\n- '.join([f for f in filtered_sentences if len(f)>0])}").send()

            # Convert Sentences to Edges
            added_edges = []

            sentences_to_add = await show_edges_to_add(filtered_sentences)
            print(sentences_to_add)

            for sentence in sentences_to_add:
                

                print(sentence)

                extracted_pos,doc = sem.extract_pos(sentence)
                head,relation,tail = sem.get_triplets(extracted_pos,doc)


                # Add Triplets to Graph
                if(head and tail):
                    heads_to_add = alias_finder.findall(head)
                    tails_to_add = alias_finder.findall(tail)

                    # Remove duplicates
                    filtered_heads = []
                    for head in heads_to_add:
                        if(not(head in filtered_heads)):
                            filtered_heads.append(head)
                    heads_to_add = filtered_heads

                    filtered_tails = []
                    for tail in tails_to_add:
                        if(not(tail in filtered_tails)):
                            filtered_tails.append(tail)
                    tails_to_add = filtered_tails

                    added_head = len(heads_to_add)>0
                    added_tail = len(tails_to_add)>0

                    if(added_head and added_tail):
                        print(heads_to_add)
                        print(tails_to_add)
                        #await cl.Message(f"Added {sentence} related to {', '.join([node for node in heads_to_add+tails_to_add])}").send()
                        [du.add_node(head_node,topic_dict[head_node]) for head_node in heads_to_add]
                        [du.add_node(tail_node,topic_dict[tail_node]) for tail_node in tails_to_add]

                        for head_node in heads_to_add:
                            for tail_node in tails_to_add:
                                key = du.add_edge(head_node,tail_node,sentence)
                                added_edges.append((head_node,tail_node,key,sentence))

                    elif(added_head):
                        print(heads_to_add)
                        #await cl.Message(f"Added {sentence} related to {', '.join([node for node in heads_to_add])}").send()
                        [du.add_node(head_node,topic_dict[head_node]) for head_node in heads_to_add]
                        for head_node in heads_to_add:
                            key = du.add_edge(head_node,head_node,sentence)
                            added_edges.append((head_node,head_node,key,sentence))
                        
                    elif(added_tail):
                        print(tails_to_add)
                        #await cl.Message(f"Added {sentence} related to {', '.join([node for node in tails_to_add])}").send()
                        [du.add_node(tail_node,topic_dict[tail_node]) for tail_node in tails_to_add]
                        for tail_node in tails_to_add:
                            key = du.add_edge(tail_node,tail_node,sentence)
                            added_edges.append((tail_node,tail_node,key,sentence))

            

            du.save_graph()

async def summarize_nodes(graph_name='graph'):

    # Summarize a node based on the descriptions

    async with cl.Step(name="Finding New Connections",icon="lightbulb") as step:
        # Find similar edges and keep one.
        edges_to_resolve = du.resolve_edges()
        du.save_graph()
        for similar_edges in edges_to_resolve:
            if(len(similar_edges)>1):
                actions=[cl.Action(name=edge[3], payload={"edge_data":edge}, label=f"{edge[0]} to {edge[1]} key {edge[2]} : {edge[3]}") for edge in similar_edges]
                response = await cl.AskActionMessage(content="Which description would you like to keep?",actions=actions,timeout=60).send()
                selected_edge = tuple(response['payload']['edge_data'])
                edges_to_delete = [edge for edge in similar_edges if not(edge==selected_edge)]
                du.knowledge_graph.remove_edges_from(edges_to_delete)

        # Find new connections between existing edges.
        new_edges = du.find_new_edges()

        for head,tail,text in new_edges:
            await cl.Message(content=f"Found a connection between {head} and {tail}!\n> {text}").send()
        du.save_graph()

async def assign_edge_labels(graph_name='graph'):

    async with cl.Step(name="Categorizing Information",icon="lightbulb") as step:

        '''
        UNUSED
        def process_wiki_schema(src_dict,topic=None):
            wiki_dict = copy.deepcopy(src_dict)
            if(topic):
                for key,value in wiki_dict.items():
                    wiki_dict[key] = wiki_dict[key].replace(f'[TOPIC]',topic)
            return wiki_dict
        '''

        for node in du.get_all_nodes():

            topic       = node['id']
            topic_type  = du.knowledge_graph.nodes[node['id']]['type']
            topic_summary = du.knowledge_graph.nodes[node['id']]['wiki']['summary']

            schema = wiki_schema[topic_type.lower()]
            labels = [f"{schema[key]}" for key in schema.keys()]
            schema_keys = list(schema.keys())

            node_wiki = dict()
            for key in schema.keys():
                node_wiki[key] = []
            
            print()
            print(node['name'])
            for edge in du.knowledge_graph.edges(node['id'],keys=True,data=True):

                edge_text = edge[3]['desc']
                results = sem.zero_shot_classification(edge_text,labels,context=[topic_summary],threshold=0.0)

                # Get best label
                best_idx = [r[1] for r in results[0]].index(max([r[1] for r in results[0]]))
                #print(best_idx)
                node_wiki[schema_keys[best_idx]].append(edge_text)
                du.set_edge_tags(edge,topic,schema_keys[best_idx])

                print(du.knowledge_graph.edges[edge[0],edge[1],edge[2]])

                #print(results)
                #print(f"{edge_text} :",[f"{r[0]} : {r[1]:.3f}" for r in results[0]])
                #print()

        du.save_graph()

async def update_node_wikis(graph_name='graph'):

    new_summaries = []
    for node in du.get_all_nodes():
        if(not(du.is_node_updated(node['id']))):

            # Update the node wikis
            unparsed_edges = du.get_unparsed_edges(node['id'])
            print(node['id'],len(unparsed_edges))

            if(len(unparsed_edges)>0):

                async with cl.Step(name=f'Updating Information about {node['name']}',icon="lightbulb") as step:

                    # Create a temporary wiki to store the information
                    temp_wiki = dict()
                    temp_wiki['summary'] = ""


                    schema = wiki_schema[du.knowledge_graph.nodes[node['id']]['type'].lower()]
                    schema_keys = list(schema.keys())

                    for key in schema.keys():
                        temp_wiki[key] = []

                    # Update the summary first
                    new_information = ' '.join([edge[3]['desc'] for edge in du.knowledge_graph.edges(node['id'],keys=True,data=True)])

                    history = []
                    props_dict = dict()
                    props_dict["node_name"]         = node['name']
                    props_dict["node_description"]  = new_information

                    history = prompts.get_node_summary_prompt(props_dict,history)
                    new_summary = await tokenize_and_generate(history,temperature=0.3,max_new_tokens=128)

                    new_summaries.append((node['id'],node['name'],new_summary,'summary'))

                    print(f'New Summary of {node['name']} : {new_summary}')
                    print()

                    # topic = node['id']

                    # Update each key in the wiki
                    for key in temp_wiki.keys():

                        wiki_information = [edge[3]['desc'] for edge in du.knowledge_graph.edges(node['id'],keys=True,data=True) if edge[3]['tags'][node['id']][0]==key]
                        print(key," : ",wiki_information)

                        if(len(wiki_information)>1):
                            props_dict = dict()
                            props_dict['node_name'] = node['name']
                            props_dict['node_description'] = " ".join(wiki_information)
                            props_dict['node_property'] = key

                            history = []
                            history = prompts.get_node_wiki_prompt(props_dict,history)
                            wiki_entry = await tokenize_and_generate(history,temperature=0.3,max_new_tokens=512)

                            new_summaries.append((node['id'],node['name'],wiki_entry,key))

                        if(len(wiki_information)==1):
                            new_summaries.append((node['id'],node['name'],wiki_information[0],key))

            else:
                print(f'{node['name']} has no new edges!')

    # Allow user to update the summaries if needed.
    summaries_to_add = await show_summaries_to_add(new_summaries)
    print(summaries_to_add)


    # CHANGE SUMMARIES TO INCLUDE THE KEY
    for node_id,key,description in summaries_to_add:
        print(node_id,key,description)
        du.set_node_wiki_description(node_id,key,description)
    
        edge_data = du.get_edges_from(node_id)
        for head,tail,key,data in edge_data:
            du.parse_edge(head,tail,key)

    #du.save_graph()

async def save_to_faiss():

    if(len(du.get_nodes_to_add_to_faiss())>0):
        await cl.Message("Would you like to save this session?",actions=[cl.Action(name='update_faiss', payload={"label": "None"}, label="Save")]).send()
    else:
        await cl.Message("No nodes have been updated!").send()

async def show_similar_edges(text,graph_name='graph'):

    edge_info = du.get_relevant_edges(text)

    if(edge_info):

        # Show all node info with actions.
        
        #props['edges'] = node_info['edge_info']

        graph_element = await update_graph_element(edge_info)

        graph_message = cl.Message(
                    content="Showing node information!",
                    elements=[graph_element]
                )

        cl.user_session.set("graph_message", graph_message)

        response = await graph_message.send()

    else:
        await cl.Message(content=f'No relevant edges found!').send()

async def show_node_info(node_name):

    node_info = du.get_node_info(node_name)

    if(node_info):
        graph_element = await update_graph_element(node_info)
        
        graph_message = cl.Message(
                    content="Showing node information!",
                    elements=[graph_element]
                )

        cl.user_session.set("graph_message", graph_message)

        response = await graph_message.send()

        

    else:
        await cl.Message(content=f'No node named {node_name} found!').send()


#############

# Corpus Analysis Tools

#############

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

# Semantic Functions

#############

async def decompose_text(text,topics=[],temperature=0.2):
    # topics : list (List of topics to focus on)
    settings = cl.user_session.get('settings')

    background_info = '\n'.join([summary for summary in du.get_node_summaries(topics) if summary])

    history = []

    props_dict = dict()
    props_dict['topics']                = '; '.join(topics)
    props_dict['text']                  = text
    props_dict['background_knowledge']  = background_info

    chat_history = prompts.get_text_decomposition_prompt(props_dict,history=[])
    print(chat_history)
    summary_response = await tokenize_and_generate(chat_history,temperature=settings['temperature'],max_new_tokens=512)
    propositions = summary_response.split('.')

    # Filter out empty strings
    propositions = ' '.join([p.strip()+'.' for p in propositions if len(p)>0])

    return propositions

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

async def brainstorm(user_topic):

    settings = cl.user_session.get('settings')
    history = cl.user_session.get("chat_history")

    if(len(history)==0):
        # Select Random Node
        if(du.check_if_node_exists(user_topic)):
            node = {"id":du.get_node_id(user_topic),"name":du.knowledge_graph.nodes[du.get_node_id(user_topic)]['name']}
        else:
            node = du.get_random_node()
        print(f"Randomly selected {node}")
        await cl.Message(content=f"Brainstorming about {node['name']}").send()

        # Use all edges coming into a single node
        #node_info = du.get_node_info(node)
        #props_dict = dict()
        #props_dict['local_context'] = '\n'.join([node_info['summary']]+[n[3] for n in node_info['edge_info']])

        # Use all neighbors of a node
        neighbors = du.graph_multihop(node['id'],2)
        print(neighbors)
        props_dict = dict()
        props_dict['topic'] = node['name']
        props_dict['local_context'] = ''
        for n in neighbors:
            node_info = du.get_node_info(n)
            props_dict['local_context'] += '\n'.join([node_info['wiki'][key] for key in node_info['wiki']])+'\n\n'


        print(props_dict['local_context'])
        # Generate an LLM Output
        history = prompts.get_brainstorming_prompt(props_dict,history)

    else:
        context_text,context_edges = await retrieve_context(user_topic)
        user_message = context_text+" "+user_topic
        history.append({"role":"user","content":user_message})

    brainstormed_json = await tokenize_and_generate(history,max_new_tokens=1024,temperature=settings['temperature'],template=ReasonedResponse)
    idea = ReasonedResponse.model_validate_json(brainstormed_json)

    actions = [cl.Action(
        name="show_reasoning",
        icon="message-circle-question-mark",
        payload={'content':idea.scratchpad},
        label="Show Reasoning"
        )]

    history.append({"role": "assistant", "content": idea.answer})
    cl.user_session.set("chat_history", history)
    print(history)

    await cl.Message(content=idea.answer,actions=actions).send()

    # Log User Response

    # Save Datapoint (Optional)

async def hypothesize(user_topic):

    settings = cl.user_session.get('settings')
    history = cl.user_session.get("chat_history")

    if(len(history)==0):
        # Select Random Node
        if(du.check_if_node_exists(user_topic)):
            node = {"id":du.get_node_id(user_topic),"name":du.knowledge_graph.nodes[du.get_node_id(user_topic)]['name']}
        else:
            node = du.get_random_node()
            print(f"Randomly selected {node}")
        await cl.Message(content=f"Hypothesizing about {node['name']}").send()

        # Use all neighbors of a node
        neighbors = du.graph_multihop(node['id'],2)

        print(neighbors[1])
        print(du.knowledge_graph.get_edge_data(node['id'],neighbors[1]))
        #summary = du.get_node_summary(node['id'])
        props_dict = dict()
        props_dict['topic'] = node['name']+", "+du.knowledge_graph.nodes[du.get_node_id(neighbors[1])]['name']

        propositions = "- "
        node_info = du.get_node_info(node['id'])
        propositions += node_info['wiki']['summary']+'\n- '
        node_info = du.get_node_info(neighbors[1])
        propositions += node_info['wiki']['summary']+'\n- '
        propositions += '\n- '.join([item['desc'] for item in du.knowledge_graph.get_edge_data(node['id'],neighbors[1]).values()])


        props_dict['facts'] = propositions#"\n- "+"\n- ".join(propositions)



        # Generate an LLM Output
        history = prompts.get_hypothesizing_prompt(props_dict,history)
    else:
        context_text,context_edges = await retrieve_context(user_topic)
        user_message = context_text+" "+user_topic
        history.append({"role":"user","content":user_message})

    print(history)
    hypotheses_json = await tokenize_and_generate(history,max_new_tokens=1024,temperature=settings['temperature'],template=HypothesisList)
    print(hypotheses_json)
    hypotheses = HypothesisList.model_validate_json(hypotheses_json)
    print(hypotheses)
    print()

    history.append({"role": "assistant", "content": hypotheses})
    cl.user_session.set("chat_history", history)

    for deduction_set in hypotheses.deductions:
        await cl.Message(content=f"{deduction_set.deduction} [BECAUSE] {deduction_set.reasoning}").send()

    # Log User Response

    # Save Datapoint (Optional)

@cl.step(name='Retrieve Context')
async def retrieve_context(topic,entity_threshold=0.25,rag_threshold=0.5,k=10):

    settings = cl.user_session.get('settings')
    #topic_with_context = await add_entity_context(topic,entity_threshold)
    context_text,context_edges = await retrieve_graph_rag(topic,threshold=settings['rag_threshold'],k=k)

    return context_text,context_edges

async def tokenize_and_generate(chat_history,max_new_tokens=256,temperature=0.6,template=None, use_chat_template=True):

    settings = cl.user_session.get("settings")

    synthesis_prompt = tokenizer.apply_chat_template(
                                chat_history,
                                tokenize=False,
                                add_generation_prompt=True
                                )

    loop = asyncio.get_running_loop()
    
    # This guarantees execution happens strictly on your single worker thread (mlx_executor)
    answer = await loop.run_in_executor(
        mlx_executor,
        _sync_generate,
        synthesis_prompt, 
        max_new_tokens, 
        settings['temperature'], 
        template
    )

    '''answer = await asyncio.to_thread(_sync_generate,
        synthesis_prompt, 
        max_new_tokens, 
        temperature, 
        template
    )'''

    '''answer = _sync_generate(
        synthesis_prompt, 
        max_new_tokens, 
        temperature, 
        template
    )'''

    return answer

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

async def check_idea_for_contradictions(message,contradiction_threshold=0.9):

    # DO NOT USE AS IS

    hypothesis_topics = du.find_topics_in_text(message.content)
    hypothesis_context = du.get_node_summaries(hypothesis_topics)

    # Decompose user text
    history = []
    history = prompts.get_decomposition_prompt(message.content,hypothesis_topics,history)
    response = await tokenize_and_generate(history,max_new_tokens=256,temperature=0.3)
    decomposition = [statement.strip() for statement in response.split('.')]
    print(response)

    # Get relevant context chains
    context_text = du.get_graph_rag_context(message.content,0.5,5)

    found_contradiction = False
    for hypothesis in decomposition:
        for premise_list in context_text:
            for premise in premise_list.split('.'):
                if(len(hypothesis)>0 and len(premise)>0):

                    premise_topics = [topic for topic in du.find_topics_in_text(message.content) if not(topic in hypothesis_topics)]
                    premise_context = du.get_node_summaries(premise_topics)
                    print(premise)

                    context = ' '.join(hypothesis_context)

                    # Get entailment scores between pairs
                    probs = sem.get_entailment_probs(premise,hypothesis,context)
                    print(probs)

                    if(probs[0]>contradiction_threshold):

                        if(not(found_contradiction)):
                            await cl.Message(content=f"Found the following lore that conflicts with your ideas!").send()
                            found_contradiction = True

                        
                        await cl.Message(content=f"{premise}").send()

                    # Keep track of relelvance of user text and chain

                    # If contradiction, respond
    
@cl.step(name='Local Context to Reason and Answer')
async def reason_and_answer(message):

    # Retrieve Base Context
    context_text,context_edges = await retrieve_context(message.content)

    props_dict = dict()
    props_dict['user_query'] = message.content
    props_dict['local_context'] = context_text

    chat_history = cl.user_session.get("chat_history")
    chat_history = prompts.get_reasoned_generation_prompt(props_dict,chat_history)

    json_answer = await tokenize_and_generate(chat_history,max_new_tokens=1024,temperature=0.3,template=ReasonedResponse)

    response = ReasonedResponse.model_validate_json(json_answer)

    actions = [cl.Action(
            name="show_reasoning",
            icon="message-circle-question-mark",
            payload={'content':response.scratchpad},
            label="Show Reasoning"
            ),
            cl.Action(
            name="show_context",
            icon="question-mark",
            payload={'context_edges':context_edges},
            label="Show Context"
            )]

    await cl.Message(content=response.answer,actions=actions).send()

#############

# MAIN APP START

#############

@cl.on_settings_update
async def on_settings_update(settings: dict):

    cl.user_session.set("settings", settings)

    print(cl.user_session.get("settings"))

@cl.on_chat_start
async def on_chat_start():

    global mlx_executor 

    for head,tail,key,edge_dict in du.knowledge_graph.edges(data=True,keys=True):
        print(du.knowledge_graph.edges[head,tail,key])
        #self.knowledge_graph.nodes[node]['updated'] = int(time.time()

    props = {"title":"",
                "subtitle":"",
                "edges":""}

    graph_element = cl.CustomElement(
                    name="GraphInfo",
                    display="side",
                    props= props
                )

    cl.user_session.set("graph_element", graph_element)
    cl.user_session.set("chat_history", [])
    
    mx.metal.clear_cache()
    mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(mlx_executor, _sync_load_models)

    settings = await cl.ChatSettings(
        [
            Slider(
                id="temperature",
                label="Creativity",
                description="Higher values encourage inventing new ideas. Lower values encourage following the given information",
                initial=0.4,
                min=0.1,
                max=0.9,
                step=0.05,
            ),

            Slider(
                id="rag_threshold",
                label="Selectiveness",
                description="Higher values encourage being more selective about the retrieved knowledge. Lower values encourage gathering more, less relevant knowledge.",
                initial=0.5,
                min=0.1,
                max=0.9,
                step=0.05,
            ),
        ]
    ).send()

    cl.user_session.set("settings", settings)

    #await create_knowledge_graph()
    #await cl.make_async(du.reload_faiss_dataset)()

    # Define System prompt
    cl.user_session.set("chat_history", [])
    cl.user_session.set("entities_to_track", [])
    cl.user_session.set("awaiting_input_edge", False)
    cl.user_session.set("awaiting_input_node", False)
    cl.user_session.set("payload", None)

    await cl.context.emitter.set_commands(helpers)
    print(du.knowledge_graph)

    await cl.Message(content="Hello! I'm Quill, your novel-writing assistant! How can I help you today?").send()

@cl.on_message
async def on_message(user_message: cl.Message):


    selected_mode = user_message.command

    if(selected_mode=='Analyze'):
        if not user_message.elements:
            await cl.Message(content="No file attached").send()
        else:
            await get_gist(user_message)
    elif(selected_mode=='Update'):

        await summarize_nodes('lore_graph')
        await assign_edge_labels('lore_graph')
        await update_node_wikis('lore_graph')
        await save_to_faiss()

        #await summarize_nodes('lore_graph')
        #du.create_dataset_from_graphs()
        #await update_datastore()

    elif(selected_mode=='Ideate'):
        #await check_idea_for_contradictions(user_message)
        await extract_knowledge_graph(user_message.content,'lore_graph','idea')
        #await remove_edge(user_message.content,'lore_graph')

    elif(selected_mode=='Add Chapter'):
        # Add a list of stories already in the user list
        story_name = 'test'
        await extract_knowledge_graph(user_message.content,story_name,'story')

    elif(selected_mode=='View'):
        #await show_similar_edges(user_message.content)
        await show_node_info(user_message.content)
    elif(selected_mode=='Brainstorm'):
        #await brainstorm(user_message.content)
        await hypothesize(user_message.content)


    elif(selected_mode=='Metadata'):
        await update_metadata(user_message)
    elif(selected_mode=='Forget'):
        cl.user_session.set("chat_history", [])
    else:
        waiting_for_node_update = cl.user_session.get("awaiting_input_node")
        waiting_for_edge_update = cl.user_session.get("awaiting_input_edge")
        if(waiting_for_node_update):
            payload = cl.user_session.get("payload")
            print(payload)
            node = payload.get("node")
            du.add_node(node,user_message.content)

            node_info = du.get_node_info(node)

            if(node_info):
                graph_element = cl.user_session.get("graph_element")
                graph_element.props['title'] = node_info['node_name']
                graph_element.props['subtitle'] = node_info['summary']
                graph_element.props['edges'] = node_info['edge_info']
                await graph_element.update()

            cl.user_session.set("awaiting_input_node", False)


        elif(waiting_for_edge_update):

            payload = cl.user_session.get("payload")
            print(payload)
            head = payload.get("head")
            tail = payload.get("tail")
            key = payload.get("key")

            du.knowledge_graph.edges[head,tail,key]['desc'] = user_message.content
            du.knowledge_graph.edges[head,tail,key]['parsed'] = False

            node_info = du.get_node_info(head)

            if(node_info):
                graph_element = cl.user_session.get("graph_element")
                graph_element.props['title'] = node_info['node_name']
                graph_element.props['subtitle'] = node_info['summary']
                graph_element.props['edges'] = node_info['edge_info']
                await graph_element.update()


            cl.user_session.set("awaiting_input_node", False)

        else:
            if not user_message.elements:
                await reason_and_answer(user_message)
                #await extract_knowledge_graph(user_message.content,'test')
                #await cl.Message(content='What are you trying to do?').send()
                
            else:
                file = user_message.elements[0]
                text = io_utils.get_text(file.path)
                resolved_text = sem.resolve_coreferences(text)
                chunks = resolved_text.split('\n\n')
                #chunks = pre.get_chunks(resolved_text,300)
     
@cl.on_chat_end
async def end_chat():
    #await update_datastore()
    pass