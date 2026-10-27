import chainlit as cl
from chainlit.input_widget import Slider
from chainlit import make_async
import json

import re
import sys
import logging
import frontmatter
import string

import utils.prompts as prompts
from utils.semantic import SemanticTools
from utils.pydantic_schema import ReasonedResponse, DecomposedText, NodeSummary, TextTuple, PhaseList, Reordering, SufficiencyCheck, ClusterThemes, TripletList, label_map, create_custom_triplet_schema
from utils.datastore_utils import DatastoreUtilities, slugify_key
import utils.pydantic_schema as py_schema

from pydantic import create_model, ValidationError
from typing import Literal
from json_repair import repair_json
from pathlib import Path

import ollama
from ollama import AsyncClient
import copy
import time

import csv

# remove later
import torch
from rapidfuzz import process, fuzz

with open('utils/tools.json', "r", encoding="utf-8") as f:
    tools = json.load(f)

with open('utils/prompts.json', "r", encoding="utf-8") as f:
    prompts_lookup = json.load(f)

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
    wiki_schema = json.load(f)

named_entities = ['character','location','artifact','faction','event','definition']

#"Qwen/Qwen2.5-0.5B-Instruct" #
punctuation_tuple = tuple(string.punctuation)

# Engineering functions (synchronous)

def update_system_prompt(system_prompt):

    #system_prompt = copy.deepcopy(prompts_lookup[prompt_name]['system'])
    system_prompt['content'] = '\n'.join(system_prompt['content'])

    chat_history = cl.user_session.get("chat_history")
    if(len(chat_history)>0):
        chat_history[0] = system_prompt
    else:
        chat_history = [system_prompt]
    cl.user_session.set("chat_history",chat_history)

def add_tool_output_to_history(tool_output):
    # tool_output = str

    chat_history = cl.user_session.get("chat_history")
    chat_history.append({"role":"tool","content":tool_output})
    cl.user_session.set("chat_history",chat_history)


def uppercase(text):
    return text[0].upper()+text[1:]

def create_node_card_dict(node,expanded=True, stack = []):

    node_card_dict = dict()

    node_card_dict['node_id'] = du.get_node_id(node)
    node_card_dict['name']  = du.get_node_name(node)
    
    node_card_dict['type']  = du.knowledge_graph.nodes[node]['type']
    node_card_dict['desc']  = du.knowledge_graph.nodes[node]['metadata']['summary']

    all_neighbors = du.get_all_neighbors(du.get_node_id(node))
    node_card_dict['neighbors']  = [(du.get_node_name(neighbor),du.get_node_type(neighbor),neighbor) for neighbor in all_neighbors['all']]

    print(node_card_dict)

    node_card_dict['expanded'] = expanded
    node_card_dict['stack'] = stack

    return node_card_dict

def add_to_stack(node, node_card_dict):

    assert 'stack' in node_card_dict.keys()

    node_card_dict['stack'].append((du.get_node_layer(node),du.get_node_name(node),node))

    return node_card_dict

async def update_graph_element(node_info):

    print(node_info)
    graph_element = cl.user_session.get("graph_element")
    graph_element.props['title'] = node_info['node_name']
    graph_element.props['type'] = node_info['node_type']
    graph_element.props['summary'] = node_info['summary']
    graph_element.props['edges'] = node_info['edge_info']
    print(graph_element.props)
    cl.user_session.set("graph_element", graph_element)
    await graph_element.update()

    return graph_element

@cl.cache
def load_modules():

    du = DatastoreUtilities(config,'worldbuilding')
    du.load_embedding_model()

    sem = SemanticTools(config)
    sem.load_extraction_model()
    #sem.load_nli_model()
    #sem.load_zsc_model()

    with open(f'../data/wikis/kiseki.json', "r", encoding="utf-8") as f:
        wiki = json.load(f)

    return du,sem,wiki

#du = DatastoreUtilities(config,'test','lore',True)
#du.load_embedding_model()
#sem = SemanticTools(config)
#sem.load_extraction_model()
du,sem,wiki = load_modules()
index = wiki.keys()
ollama_client = AsyncClient()

#############

# Testing Functions for Existing Wikis. Remove later.

#############

def filter_wiki_matches(results):

    filtered_results = dict()

    for match, score, index in results:
        if(not('/Gallery' in match) and not('/Gameplay' in match)):
            cleaned_text = re.split(r"see also", wiki[match]['content'], flags=re.IGNORECASE)[0].strip()
            cleaned_text = re.split(r"See also", cleaned_text, flags=re.IGNORECASE)[0].strip()
            cleaned_text = re.split(r"References Sources", cleaned_text, flags=re.IGNORECASE)[0].strip()
            cleaned_text = re.sub(r"\|-\|\s*([^=]+)=", r"\n\1:", cleaned_text)
            cleaned_text = cleaned_text.replace("Official Website Page (EN-JP)","")
            filtered_results[match] = cleaned_text

    return filtered_results

async def find_in_wiki(query):

    results = process.extract(query, index, limit=5)
    results = filter_wiki_matches(results)

    results_keys = list(results.keys())+['Skip']
    actions = [cl.Action(name=key, payload={"selected":key}, label=key) for key in results_keys]
    wiki_msg = cl.AskActionMessage(content="Please select which topic you'd like to analyze!",actions=actions)
    response = await wiki_msg.send()

    if(not(response['payload']['selected']=='Skip')):
        selection = response['payload']['selected']
        wiki_msg.content = f"Set input text to the following:\n### {selection}\n{results[selection]}"
        await wiki_msg.update()
        cl.user_session.set("wiki_text",results[selection])
    else:
        cl.user_session.set("wiki_text","")

#############

# Callbacks

#############

@cl.action_callback("change_node")
async def on_change_node(action: cl.Action):

    node_card_element = cl.user_session.get("node_card_element")
    current_stack = node_card_element.props['stack']
    payload = action.payload
    node_id = payload.get("node_id")

    node_card_dict = create_node_card_dict(node_id,stack=current_stack)
    node_card_element = cl.user_session.get("node_card_element")
    node_card_element.props = node_card_dict
    cl.user_session.set("node_card_element",node_card_element)
    await node_card_element.update()

@cl.action_callback("exit")
async def on_exit(action: cl.Action):
    payload = action.payload
    save = payload.get("save")
    node_card_element = cl.user_session.get("node_card_element")

    if(save):
        new_desc = payload.get("new_desc")
        new_layer = payload.get("new_layer")
        node_id = node_card_element.props['node_id']
        
        # Add function to update nodes.
        # Update summary / desc + Extrapolate if needed
        # Update layer + Update all related edges. Show error if there is a problem.
        # Add edges between each consecutive pair in stack

        await update_node_description(node_id,new_desc)

        await update_node_layer(node_id,new_layer)

        du.reload_graph()
    else:
        du.reload_graph()
        node_card_element.props['stack'] = []
        cl.user_session.set("node_card_element",node_card_element)

        node_card_message = cl.user_session.get("node_card_message")
        node_card_message.elements = []
        node_card_message.content = "Closing information panel!"
        await node_card_message.remove()

@cl.action_callback("stack")
async def on_stack(action: cl.Action):

    node_id = action.payload.get("node_id")
    idx = action.payload.get("idx")
    remove = action.payload.get("rem")

    print(node_id,idx,remove)

    if(remove):
        pass
    else:
        node_card_element = cl.user_session.get("node_card_element")
        node_card_dict = node_card_element.props
        node_card_dict = add_to_stack(node_id, node_card_dict)
        node_card_element.props = node_card_dict
        cl.user_session.set("node_card_element",node_card_element)
        await node_card_element.update()

    # Add or remove the relevant node
    #node_card_dict = create_node_card_dict(node_id)
    #node_card_message = cl.user_session.get("node_card_message")
    #node_card_message.elements.append(node_card_element_side)
    #node_card_message.content = "Adding card to the side!"
    #await node_card_message.update()

#############

# Datastore Operations

#############

@cl.action_callback("update_faiss")
async def update_datastore(action: cl.Action):

    msg = cl.Message(content="Updating Knowledge Base...")

    await msg.send()

    dataset,index = await make_async(du.update_faiss_dataset)()
    await make_async(du.overwrite_faiss_dataset)(dataset,index)

    msg.content = "Knowledge Base Updated!"

    await msg.update()

async def save_to_faiss():

    not_in_graph,not_in_dataset,nodes_with_unsaved_changes = du.calculate_graph_datastore_diff(write_to_kg2ds=False)

    graph_diff = len(not_in_graph)>0
    dataset_diff = len(not_in_dataset)>0
    saved_diff = len(nodes_with_unsaved_changes)>0

    if(graph_diff or dataset_diff or saved_diff):
        await cl.Message("Would you like to save this session?",actions=[cl.Action(name='update_faiss', payload={"label": "None"}, label="Save")]).send()
    else:
        await cl.Message("No nodes have been updated!").send()

#############

# Visualization Functions

#############

async def show_knowledge_graph(user_message):

    node = du.get_relevant_topics(user_message,threshold=0.8,all=False)
    if(node):
        #nodes,edges = du.get_node_stack(node)
        nodes = [node]
        print("Nodes here : ",nodes)
        print(nodes)
        actions = [cl.Action(name='add_modify',payload={'modify':'node'},label='Add/Modify Node'),cl.Action(name='add_modify',payload={'modify':'edge'},label='Add/Modify Connection')]
        fig = du.create_network_figure(nodes)
        element = cl.Plotly(name="network_graph", figure=fig, display="inline")
        msg = await cl.Message(content="Hover over the nodes or edges for more information!",elements=[element],actions=actions).send()
    else:
        response = await cl.Message("No relevant nodes found!").send()
    
async def show_checklist(entities,message='Select topics.',show_description=True,show_response=True):

    # Entities : Dict: entity:label (e.g "Alvar" : "character")

    items_list = []
    for key in entities.keys():

        entity = key
        label = entities[key]

        items_list.append({
                            "id":str(key),
                            "label":entities[key]['primary_alias'],
                            "description":entities[key]['existing_aliases'],
                            "defaultChecked": True
                        })
        
    print(items_list)
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

    print(selection_response)
    selections = []
    new_entities = dict()
    for key in selection_response.keys():
        if(not(key=='submitted')):
            if(selection_response[key][2]):
                selections.append(int(key))
        
            new_entities[int(key)] = dict()
            new_entities[int(key)]['primary_alias'] = selection_response[key][0]
            new_entities[int(key)]['existing_aliases'] = list(set(selection_response[key][1]+[selection_response[key][0]]))
            new_entities[int(key)]['alias_known'] = selection_response[key][3]
            new_entities[int(key)]['to_add'] = selection_response[key][2]


    return new_entities

async def show_edges_to_add(sentence_tuples,message='Select topics.',top_text="",show_description=True):
    # sentence_tuples list(tuple) : List of (sentence,label,topic) tuples

    items_list = []
    for idx,tuplet in enumerate(sentence_tuples):
        #items_list.append({"id":idx,"text":tuplet[0],"label":tuplet[1],"topics":tuplet[2]})
        items_list.append({"id":idx,"text":tuplet[0],"label":tuplet[1]})

    
    props = {
            "timeout": 6000,
            "topText": top_text,
            "Title": message,
            "items": items_list,
            "options":[label_map[schema] for schema in TextTuple.model_json_schema()['properties']['label']['enum']]}

    checklist_element = cl.CustomElement(
        name="AtomicSentenceList",
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
                # Return (text,label,topics)
                edges_to_add.append([selection_response[key][0],selection_response[key][3]])

    await element_msg.remove()

    return edges_to_add

async def show_information(user_text):

    node = du.get_mentioned_topic(user_text)

    if(node):
        node_card_dict = create_node_card_dict(node)
        node_card_dict['stack'] = [] # This is crucial. Idk why.
        node_card_element = cl.user_session.get("node_card_element")
        node_card_element.props = node_card_dict
        cl.user_session.set("node_card_element",node_card_element)
        node_card_message = cl.Message(content=f"Showing information about {du.get_node_name(node)}!",elements=[node_card_element])
        cl.user_session.set("node_card_message",node_card_message)
        await node_card_message.send()

    else:
        await cl.Message(content=f"No information to show about {user_text}!").send()

#############

# Composite Utilities

#############

async def resolve_coreferences(user_text,choose=True):

    entities = sem.extract_named_entities(user_text)
    entity_dict = sem.merge_existing_aliases(entities,du.inverse_lookup)

    resolved_text = sem.resolve_aliases(user_text,entity_dict,True)

    print(entity_dict)

    return resolved_text, entity_dict    

async def extract_topics(og_text):

    resolved_text, entity_dict = await resolve_coreferences(og_text)

    # Use an LLM to find entities that the NER may have missed.
    props_dict = dict()
    props_dict['statements']            = resolved_text
    props_dict['known_topics']          = "\n".join([entity_dict[key]['primary_alias'] for key in entity_dict.keys()])

    # Clear out chat history to avoid cross-contamination
    chat_history = prompts.prompt_parser("entity_discovery_prompt",props_dict)

    llm_entity_discovery_response = await tokenize_and_generate(temperature=0.3,max_new_tokens=1024,template=py_schema.EntityDiscoveryList)

    curr_key = len(entity_dict.keys())+1
    for info in llm_entity_discovery_response.entities_discovered:
        
        print(f"{info.entity_name} ({info.entity_type} - {info.entity_subcategory}) : {info.reasoning}")

        # Add the new entities into the entity_dict for adding.
        ner_discovered_entities = [entity_dict[key]['primary_alias'] for key in entity_dict.keys()]
        if(not(info.entity_name in ner_discovered_entities)):
            entity_dict[curr_key] = dict()
            entity_dict[curr_key]['primary_alias'] = info.entity_name
            entity_dict[curr_key]['existing_aliases'] = [info.entity_name]
            entity_dict[curr_key]['alias_known'] = False
            entity_dict[curr_key]['to_add'] = False
            curr_key += 1

    entity_dict = await show_checklist(entity_dict,message='Please select the topics to add to the story graph!')

    nodes_to_add = cl.user_session.get("relevant_nodes")

    # Add all the new "topic" nodes to the graph and reload the maps
    for key in entity_dict.keys():
        print(entity_dict[key])
        if(entity_dict[key]['to_add'] and not(du.get_node_id(entity_dict[key]['primary_alias']))):

            if(not(entity_dict[key]['alias_known']) and len(entity_dict[key]['existing_aliases'])>1):
                # Prompt user to provide an alias
                actions = [cl.Action(name=alias,payload={'selection':alias},label=alias) for alias in entity_dict[key]['existing_aliases']+['Other','Skip']]
                response = await cl.AskActionMessage(content=f"Please provide the primary alias for {entity_dict[key]['primary_alias']}!",actions=actions).send()

                if(response['payload']['selection']=='Other'):
                    user_response = await cl.AskUserMessage(content=f"Please type in the new alias!").send()
                    entity_dict[key]['primary_alias'] = response['output'].strip()
                    entity_dict[key]['existing_aliases'] += [response['output'].strip()]
                    entity_dict[key]['existing_aliases'] = list(set(entity_dict[key]['existing_aliases']))

                elif(not(response['payload']['selection']=='Skip')):
                    entity_dict[key]['primary_alias'] = response['payload']['selection']

            props_dict = dict()
            props_dict['statements']            = resolved_text
            props_dict['topic']                 = entity_dict[key]['primary_alias']

            props_dict['background']            = ""
            if(du.get_node_id(entity_dict[key]['primary_alias'])):
                props_dict['background']            = du.knowledge_graph.nodes[du.get_node_id(entity_dict[key]['primary_alias'])]['metadata']['summary']

            if(len(props_dict['background'].strip())==0):
                props_dict['background']            = f"No background knowledge about {entity_dict[key]['primary_alias']} available."


            # Clear out chat history to avoid cross-contamination
            chat_history = prompts.prompt_parser("entity_categorization_prompt",props_dict)
            
            llm_entity_category_response = await tokenize_and_generate(temperature=0.3,max_new_tokens=128,template=py_schema.EntityType)

            await cl.Message(content=f"{entity_dict[key]['primary_alias']}\n{llm_entity_category_response.entity_type.value} ({llm_entity_category_response.entity_subcategory}): {llm_entity_category_response.reasoning}").send()
            print()

            if(du.get_node_id(entity_dict[key]['primary_alias'])):

                existing_id = du.get_node_id(entity_dict[key]['primary_alias'])

                existing_type = du.get_node_type(existing_id)
                new_type = llm_entity_category_response.entity_type.value

                if(not(existing_type==new_type)):
                    options = [cl.Action(name=option.lower(), payload={"selected":option.lower()}, label=option) for option in ["Keep Old","Keep New"]]
                    response = await cl.AskActionMessage(content=f"Existing type for {entity_dict[key]['primary_alias']} is '{existing_type}'. Discovered type is '{new_type}'. Which one should I keep?",actions=options).send()

                    if(response['payload']['selected']=='keep old'):
                        new_node_type = existing_type
                    else:
                        new_node_type = new_type
                else:
                    new_node_type = existing_type

                new_node_name = entity_dict[key]['primary_alias']
                new_node_tag = llm_entity_category_response.entity_subcategory
                new_aliases = entity_dict[key]['existing_aliases']

            else:

                new_node_name = entity_dict[key]['primary_alias']
                new_node_type = llm_entity_category_response.entity_type.value
                new_node_tag = llm_entity_category_response.entity_subcategory
                new_aliases = entity_dict[key]['existing_aliases']

            nodes_to_add.append(du.add_node(new_node_name,
                                type=new_node_type,
                                tag=new_node_tag,
                                aliases=new_aliases))

    du.reload_lookups()   

    cl.user_session.set("relevant_nodes",nodes_to_add) 

    return resolved_text, entity_dict

async def extract_relationships(resolved_text,entity_dicts):

    nodes_to_add = cl.user_session.get("relevant_nodes") 

    # Combine all names
    all_aliases = [alias for idx,entity_dict in entity_dicts.items() for alias in entity_dict['existing_aliases']]

    topics = [entity_dict['primary_alias'] for entity_dict in entity_dicts.values() if entity_dict['to_add']]+['Other']

    # 3. Create a new model that inherits from BaseReport and adds/overrides fields
    triplet_schema = create_custom_triplet_schema(topics)

    # Split text into chunks here
    chunks = sem.split_into_chunks(resolved_text)
    
    new_edges = []

    for chunk in chunks:

        # Filter Chunk to only consider sentences where the entity is mentioned
        statements = sem.split_into_sentences(chunk)
        triplet_widget_props = await decompose_text(statements,topics,template=triplet_schema)

        # Check relationships to save.
        triplet_widget_element = cl.CustomElement(name="TripletListWidget",display="inline",props= triplet_widget_props)
        print(triplet_widget_element.props)
        response = await cl.AskElementMessage(content="Select Relationships to save!",element=triplet_widget_element).send()
        print('Response : ',response)
        # response[key] = [heads,relation,tails,text,selected,selected_heads,selected_tails)
        # Reification of Edges

        all_relations = []
        if(response):
            for key in response.keys():
                if(not(key=='submitted') and response[key]['selected']):

                    heads               = [response[key]['head'][idx] for idx in [int(key) for key,value in response[key]['head_selection'].items() if value]]
                    relation            = response[key]['relation']
                    tails               = [response[key]['tail'][idx] for idx in [int(key) for key,value in response[key]['tail_selection'].items() if value]]
                    text                = response[key]['base_fact']
                    to_add              = response[key]['selected']
                    temporal_metadata   = response[key]['temporal_metadata']


                    if(len(heads)>1 or len(tails)>1):

                        # 1. Convert edge to a node and add it
                        await cl.Message(content="Reifying Edge!").send()

                        topics_mentioned = heads+tails

                        # 1.A. Reify edge into a node and get the title, type and tags.
                        props_dict = dict()
                        props_dict['statement']            = text
                        props_dict['background']            = ""
                        for topic in topics_mentioned:
                            props_dict['background']            += "* "+du.knowledge_graph.nodes[du.get_node_id(topic)]['metadata']['summary']+'\n'

                        # Clear out chat history to avoid cross-contamination
                        chat_history = prompts.prompt_parser("edge_reification_prompt",props_dict)
                    
                        llm_reification_response = await tokenize_and_generate(temperature=0.25,max_new_tokens=128,template=py_schema.ReifiedEdge)

                        new_node_name = llm_reification_response.title
                        new_node_type = llm_reification_response.entity_type.value
                        new_node_tag = llm_reification_response.entity_subcategory
                        new_aliases = [llm_reification_response.title]
                        new_node_summary = text

                        nodes_to_add.append(du.add_node(new_node_name,
                                            type=new_node_type,
                                            tag=new_node_tag,
                                            aliases=new_aliases,
                                            summary=new_node_summary))

                        du.reload_lookups()

                        await cl.Message(content=f"Created a new topic : **{new_node_name}** ({llm_reification_response.entity_type.value})\n{text}").send()

                        # Classify new edges from each node to the new node. 

                        reification_statement = [text]

                        all_new_triplets = []
                        for topic in heads:
                            reification_schema = create_custom_triplet_schema([topic,new_node_name],exclusive=True)
                            reification_triplets = await decompose_text(reification_statement,[topic,new_node_name],template=reification_schema,exclusive=True)
                            all_new_triplets += reification_triplets['triplets']

                        for topic in tails:
                            reification_schema = create_custom_triplet_schema([new_node_name,topic],exclusive=True)
                            reification_triplets = await decompose_text(reification_statement,[new_node_name,topic],template=reification_schema,exclusive=True)
                            all_new_triplets += reification_triplets['triplets']

                        for triplet in all_new_triplets:
                            reification_relation = dict()
                            reification_relation['head']                = triplet['head'][0]
                            reification_relation['relation']            = triplet['relation']
                            reification_relation['tail']                = triplet['tail'][0]
                            reification_relation['text']                = triplet['base_fact']
                            reification_relation['temporal_metadata']   = triplet['temporal_metadata']

                            await cl.Message(content=f"Connection between **{reification_relation['head']}** and **{reification_relation['tail']}** {reification_relation['text']}").send()

                            all_relations.append(reification_relation)

                    else:
                        curr_relation = dict()
                        curr_relation['head']               = heads[0]
                        curr_relation['relation']           = relation
                        curr_relation['tail']               = tails[0]
                        curr_relation['text']               = text
                        curr_relation['temporal_metadata']  = temporal_metadata

                        all_relations.append(curr_relation)


        if(len(all_relations)>0):
            for relation_info in all_relations:

                props_dict = dict()
                props_dict['statement']            = f"{relation_info['text']} : Relationship between {relation_info['head']} and {relation_info['tail']}"

                props_dict['background']            = []
                for node in [relation_info['head'],relation_info['tail']]:
                    if(du.get_node_id(node)):
                        props_dict['background'].append(du.knowledge_graph.nodes[du.get_node_id(node)]['metadata']['summary'])
                    else:
                        props_dict['background'].append(f"No background knowledge about {node} available.")

                props_dict['background'] = "\n".join(props_dict['background'])

                # Clear out chat history to avoid cross-contamination
                chat_history = prompts.prompt_parser("relationship_categorization_prompt",props_dict)
                
                llm_relation_category_response = await tokenize_and_generate(temperature=0.3,max_new_tokens=128,template=py_schema.RelationshipType)

                await cl.Message(content=f"{relation_info['text']}\n{llm_relation_category_response.relation_type.value} ({llm_relation_category_response.relation_subcategory}): {llm_relation_category_response.reasoning}").send()
                print()

                # Add edge here

                if(not(relation_info['tail'].lower()=='other' and relation_info['head'].lower()=='other')):

                    if(relation_info['tail'].lower()=='other'):
                        relation_info['tail'] = relation_info['head']
                    elif(relation_info['head'].lower()=='other'):
                        relation_info['head'] = relation_info['tail']

                    # Check if edges between nodes are redundant
                    if(not(du.check_edges_for_redundancy(relation_info['head'],relation_info['tail'],relation_info['text']))):
                        new_edges.append(du.add_edge(relation_info['head'],relation_info['tail'],
                                                    type=llm_relation_category_response.relation_type.value,
                                                    tag=llm_relation_category_response.relation_subcategory,
                                                    text=relation_info['text'],
                                                    temporal_data=relation_info['temporal_metadata']))
                    else:
                        await cl.Message(content=f"{relation_info['text']} is similar to information already known!").send()

    cl.user_session.set("relevant_edges",new_edges)
    cl.user_session.set("relevant_nodes",nodes_to_add)
    #du.save_graph()


    return nodes_to_add

async def decomposition_post_processing(nodes_to_add, entity_dicts):

    relevant_nodes = cl.user_session.get("relevant_nodes")

    # Check against existing nodes for similarity and remove redundant nodes here.
    async with cl.Step(name="Information Redundancy Check",default_open=True) as step:
        #to_compare = [text for text,type in nodes_to_add]
        #comparison_dict = du.compare_against_existing_nodes(to_compare)
        comparison_dict = dict()

        existing_nodes_to_remove = []
        new_nodes_to_not_add = []
        for idx,similar_nodes in comparison_dict.items():

            if(len(similar_nodes)>0):
                for node in similar_nodes:
                    print(f"{to_compare[idx]} --> {du.get_node_name(node)}")
                    actions = [
                                cl.Action(name="new", payload={"selected":'new'}, label=f"NEW : {to_compare[idx]}"),
                                cl.Action(name="new", payload={"selected":'old'}, label=f"OLD : {du.get_node_name(node)}"),
                                cl.Action(name="both", payload={"selected":'both'}, label="BOTH")
                                ]

                    response = await cl.AskActionMessage(content=f"Found a node similar to **{to_compare[idx]}**! Which one should I keep?",actions=actions).send()
                    if(response['payload']['selected']=='new'):
                        existing_nodes_to_remove.append(node)
                    elif(response['payload']['selected']=='old'):
                        new_nodes_to_not_add.append(idx)

        # Remove the nodes
        new_nodes_to_not_add = sorted(new_nodes_to_not_add,reverse=True)

        for idx in new_nodes_to_not_add:
            nodes_to_add.pop(idx)

        for node_to_remove in existing_nodes_to_remove:
            du.remove_node(node_to_remove)

        # Add nodes
        new_nodes = []
        new_edges = []
        print(f"\nNodes to Add : {list(set(nodes_to_add))}")

    du.save_graph()

    #cl.user_session.set("relevant_nodes",relevant_nodes)

    return new_nodes,new_edges

#############

# Learning from Input Text

#############

async def decompose_text(statements,topics,template=None,temperature=0.2,exclusive=False):
    # statements = list(str)
    # topics = list(str)

    async with cl.Step(name=f"Triplet Extraction",default_open=True) as step:
        if(template==None):
            template = TripletList

        # Extract statements and topics
        background_info = []
        for topic in topics:
            if(du.get_node_id(topic) and len(du.knowledge_graph.nodes[du.get_node_id(topic)]['metadata']['summary'])>0):
                background_info.append(f"{topic} : {du.knowledge_graph.nodes[du.get_node_id(topic)]['metadata']['summary']}")

        # Clear out chat history to avoid cross-contamination
        if(exclusive):
            props_dict = dict()
            props_dict['background']            = '\n\n'.join(background_info)
            props_dict['topics']                = f"* source topic : {topics[0]}\n* target topic : {topics[1]}"
            props_dict['statements']            = '\n\n'.join([f"<statement>\n{statement}\n</statement>" for statement in statements])
            print(props_dict)
            chat_history = prompts.prompt_parser("triplet_extraction_prompt_force_exclusive",props_dict)

        else:
            props_dict = dict()
            props_dict['background']            = '\n\n'.join(background_info)
            props_dict['topics']                = '\n'.join([f"* {topic}" for topic in topics])
            props_dict['statements']            = '\n\n'.join([f"<statement>\n{statement}\n</statement>" for statement in statements])
            print(props_dict)
            chat_history = prompts.prompt_parser("triplet_extraction_prompt",props_dict)
        
        settings = cl.user_session.get('settings')

        llm_response = await tokenize_and_generate(temperature=0.3,max_new_tokens=2048,template=template)

        triplet_widget_props = {'triplets':[]}
        for idx,triplet in enumerate(llm_response.triplets):
            print(f"{triplet.simplified_text} : {triplet.sources} --> {triplet.relation} --> {triplet.targets} @ {triplet.temporal_metadata} ({triplet.reasoning})")

            triplet_data = dict()

            triplet_data['id'] = idx
            triplet_data['head'] = triplet.sources
            triplet_data['relation'] = triplet.relation
            triplet_data['tail'] = triplet.targets
            triplet_data['base_fact'] = triplet.simplified_text
            triplet_data['temporal_metadata'] = triplet.temporal_metadata

            triplet_widget_props['triplets'].append(triplet_data)

    return triplet_widget_props

async def extract_knowledge_graph(og_text,graph_name='graph',graph_type='story'):

    if(graph_type=='idea'):
        graph_name = 'lore_graph'

    async with cl.Step(name="Topic Categorization",default_open=True) as step:
        # Extract (Named Entities from Text + Nodes from Graph + Additional User Specified Entities)
        resolved_text, entity_dict = await extract_topics(og_text)

    # Summarize the scenes
    #async with cl.Step(name="Extracting Information",default_open=True) as step:
    async with cl.Step(name="Relationship Extraction",default_open=True) as step:
        nodes_to_add = await extract_relationships(resolved_text, entity_dict)

    # Postprocessing

    relevant_nodes = cl.user_session.get("relevant_nodes")
    fig = du.create_network_figure(relevant_nodes)
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    msg = await cl.Message(content="Updated the knowledge graph!",elements=[element]).send()
    cl.user_session.set("relevant_nodes",[])

async def summarize_nodes():

    updated_nodes = []
    for node in du.knowledge_graph.nodes():

        # Check if the node needs to be updated
        if(du.knowledge_graph.nodes[node]['metadata']['summary_updated']<du.knowledge_graph.nodes[node]['updated']):

            node_name = du.get_node_name(node)
            summary_content = f"## {node_name}"
            summary_message = cl.Message(content=summary_content)
            await summary_message.send()

            edge_data = du.get_all_edges(node)

            node_information = [f"Name : {node_name}\nCategory : ({du.get_node_type(node)})\nTags : {", ".join(du.get_node_tags(node))}\n"]
            for key,data in edge_data:
                if('tag' in data.keys()):
                    node_information.append(f"{data['type']} ({data['tag']}) : {data['text']}")
                else:
                    node_information.append(f"{data['type']} ({", ".join(data['tags'])}) : {data['text']}")

            node_information = "\n".join(node_information)

            args_dict = dict()
            args_dict['node_name'] = node_name
            args_dict['information'] = node_information

            chat_history = prompts.prompt_parser("node_definition_prompt",args_dict)
            
            response = await tokenize_and_generate(max_new_tokens=512,temperature=0.3)

            summary_content = f"## {node_name}\n> {response.message.content}"
            summary_message.content = summary_content
            await summary_message.update()

            du.add_metadata(node,replace=True,summary=response.message.content)

    du.save_graph()

    #fig = du.create_network_figure(updated_nodes,target_edges=[])
    #element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    #msg = await cl.Message(content="Updated the highlighted nodes!",elements=[element]).send()
        
async def add_node_alias(node_name):
    if(du.does_node_exist(node_name)):

        response = await cl.AskUserMessage(content=f"Please provide an alias for {node_name}!", timeout=30).send()
        alias = response['output'].strip()
        aliases = du.add_alias(node_name,alias)
        await cl.Message(content=f"Aliases for {node_name} are {", ".join(aliases)}").send()
    else:
        await cl.Message(content=f"No topic named '{node_name}'' found!").send()

async def update_node_description(node_name, new_description):

    node_card_message = cl.user_session.get("node_card_message")

    if(len(new_description.strip())>0 and not(new_description==du.get_node_name(node_name))):
        node_id = du.get_node_id(node_name)
        du.update_node_description(node_id,new_description)

        # If node is plot
        if(du.get_node_layer(node_id)=='plot'):
            # topic --> plot
            new_edges = added_edges = du.extrapolate_for_node(node_id,'involvement')

        # If node is lore
        if(du.get_node_layer(node_id)=='lore'):
            # topic --> lore
            new_edges = added_edges = du.extrapolate_for_node(node_id,'description')


        # If node is topic, there is no change because the extrapolation happens because of the other nodes
        node_card_message.content=f"Updated the description to {du.get_node_name(node_name)}"
        await node_card_message.update()
        print(new_edges)
        du.save_graph()
        du.reload_graph()

async def update_node_layer(node_name,new_layer):
    # Only allow transitions from plot to lore and vice versa.

    node_card_message = cl.user_session.get("node_card_message")
    print(new_layer)
    if(not(new_layer==du.get_node_layer(node_name)) and not(new_layer=='topic' or du.get_node_layer(node_name)=='topic')):
        new_layer = new_layer.lower()
        du.update_node_layer(node_name,new_layer)
        du.save_graph()
        du.reload_graph()

        node_card_message.content=f"Changed the layer to {uppercase(new_layer)}"
        await node_card_message.update()

async def reorder_plot(user_query):

    # Add timeline widget

    props = {'steps':[]}
    timeline_widget = cl.CustomElement(
        name="TimelineWidget",
        props=props
    )

    timeline_message = cl.Message(content="Finding Events!",elements=[timeline_widget])
    await timeline_message.send()

    og_text,_ = du.resolve_aliases(user_query)
    existing_alias_dict = du.get_alias_dict(og_text)

    print(f"\nInitial Resolution : {og_text}")
    print(f"\nExisting Entity Dict : {list(existing_alias_dict.keys())}")

    doc,entity_coreferences = sem.get_alias_clusters(og_text,existing_alias_dict)
    entity_dict,replacements = sem.match_clusters_against_aliases(doc,entity_coreferences,existing_alias_dict)
    resolved_text = sem.resolve_aliases(doc,replacements)

    print(f"\nFinal Resolved Text : {resolved_text}")

    clauses = []
    for sentence in sem.split_into_sentences(resolved_text):
        clauses += sem.advanced_split_and_repair(sentence)

    print(f"\advanced_split_and_repair : {resolved_text} ---> {clauses}")
    nodes = du.find_nodes_in_clauses(clauses,threshold=0.8,filter='plot')

    nodes = list(set(nodes))

    ordered_nodes = []

    current_step = 1
    while(len(nodes)>0):

        context, node_dict = du.get_plot_to_reorder(nodes)
        timeline_message.content = "Establishing Chronology for the following events!\n"+"\n".join([f"• **{du.get_node_name(node_dict[node])}**" for node in node_dict.keys()])
        await timeline_message.update()

        cl.user_session.set("chat_history", [])
        
        args_dict = dict()
        args_dict['phases'] = context
        args_dict['outline'] = resolved_text
        history = prompts.prompt_parser("plot_reordering_prompt",args_dict)
        
        response = await tokenize_and_generate(max_new_tokens=128,temperature=0.0,template=PhaseList)

        event_id = response.event_id
        print(response.thinking)

        if(not(event_id in node_dict.keys())):
            break
        else:
            #await cl.Message(content=f"{current_step}. {du.get_node_name(node_dict[event_id])}").send()
            timeline_widget.props['steps'].append({"id":current_step,"title":du.get_node_name(node_dict[event_id]),"subtitle":f"Event {current_step}"})
            await timeline_widget.update()

            current_step += 1
            ordered_nodes.append(node_dict[event_id])
            node_dict.pop(event_id)
            nodes = list(node_dict.values())

    timeline_message.content = "Ordered Plot in Chronological Order!"
    await timeline_message.update()

    added_edges, edges_to_reconsider = du.order_scenes(ordered_nodes)

    target_edges = []


    if(len(edges_to_reconsider['full_overlap'])>0):
        for overlap_dict in edges_to_reconsider['full_overlap']:
            focus_on = dict()
            focus_on['nodes'] = overlap_dict['common_nodes']
            focus_on['edges'] = overlap_dict['common_edges']
            fig = du.create_network_figure(overlap_dict['chapter_nodes'],target_edges = overlap_dict['chapter_edges'], add_edges=True, focus_on=focus_on)
            element = cl.Plotly(name="network_graph", figure=fig, display="inline")
            msg = await cl.Message(content="Please remove one of the following edges first!",elements=[element]).send()

    else:
        for overlap_dict in edges_to_reconsider['least_cut']:

            focus_on = dict()
            focus_on['nodes'] = list(set([node for node in overlap_dict['option_1']]+[node for node in overlap_dict['option_2']]))
            focus_on['edges'] = [overlap_dict['option_1'],overlap_dict['option_2']]
            fig = du.create_network_figure(overlap_dict['nodes_in_path'],add_edges=True, focus_on=focus_on)
            element = cl.Plotly(name="network_graph", figure=fig, display="inline")
            msg = await cl.Message(content="",elements=[element]).send()

            options = dict()
            options['option_1'] = f"{du.get_node_name(overlap_dict['option_1'][0])}  ►  {du.get_node_name(overlap_dict['option_1'][1])}"
            options['option_2'] = f"{du.get_node_name(overlap_dict['option_2'][0])}  ►  {du.get_node_name(overlap_dict['option_2'][1])}"
            options['option_3'] = "Don't add the new edge."

            actions = [cl.Action(name=key,payload={'selection':key.lower()},label=text) for key,text in options.items()]
            response = await cl.AskActionMessage(content="I need to remove a connection to maintain the structure of the plot!\nWhich connection should I remove?",actions=actions).send()
            if(response['payload']['selection']=='option_1'):
                # Remove Least cut 1 edge
                removed_edge = du.remove_edge(overlap_dict['option_1'][0],overlap_dict['option_1'][1])
                du.add_edge(overlap_dict['new_edge'][0],overlap_dict['new_edge'][1],"scene")
                target_edges.append(overlap_dict['new_edge'][0],overlap_dict['new_edge'][1])

            if(response['payload']['selection']=='option_2'):
                # Remove Least cut 2 edge
                removed_edge = du.remove_edge(overlap_dict['option_2'][0],overlap_dict['option_2'][1])
                du.add_edge(overlap_dict['new_edge'][0],overlap_dict['new_edge'][1],"scene")
                target_edges.append((overlap_dict['new_edge'][0],overlap_dict['new_edge'][1]))

        for head,tail,desc in added_edges:
            target_edges.append((head,tail))

        fig = du.create_network_figure(ordered_nodes,target_edges=target_edges)
        element = cl.Plotly(name="network_graph", figure=fig, display="inline")
        actions = [cl.Action(name=name,payload={'selection':name.lower()},label=name) for name in ['Yes','No']]
        msg = await cl.Message(content="Updated the knowledge graph!",elements=[element]).send()

        action_msg = cl.AskActionMessage(content="Should I save it?",actions=actions)
        response = await action_msg.send()
        
        if(response['payload']['selection']=='yes'):
            du.save_graph()
            action_msg.content = "Saved the Graph!"
            await action_msg.update()
        else:
            action_msg.content = "Did NOT save the Graph!"
            await action_msg.update()

#############

# User Query Answering

#############

async def choose_and_use_tool(user_message):

    settings = cl.user_session.get("settings")

    # Extract Topics...

    resolved_text,entity_dict = await resolve_coreferences(user_message.content,choose=False)

    relevant_nodes = du.get_relevant_topics(resolved_text,threshold=0.6,all=True)
    mentioned_topics = [du.get_node_name(node) for node in relevant_nodes]
    topic_summaries = [du.knowledge_graph.nodes[node]['metadata']['summary'] for node in relevant_nodes]

    print("Mentioned : ",mentioned_topics)
    print("Resolved : ",resolved_text)

    #chat_history.append({'role': 'user', 'content': resolved_text})

    # Devising Strategy...
    args_dict = dict()
    args_dict['query'] = resolved_text+'\n\n'+'\n'.join(topic_summaries)
    args_dict['topics'] = ", ".join(mentioned_topics)

    chat_history = prompts.prompt_parser("pivot_node_strategy_prompt",args_dict,refresh=False)
    
    print(chat_history[-1]['content'])

    llm_response = await tokenize_and_generate(max_new_tokens=512,temperature=0.3,template=py_schema.TopicIdentifierList)

    exploration_strategy = dict()
    for topic_info in llm_response.topics_identified:
        print(topic_info)
        exploration_strategy[topic_info.topic] = [topic.value for topic in topic_info.selected_information_categories]

    print(exploration_strategy)

    gathered_information = await explore_knowledge_graph(exploration_strategy)

    prompts.add_to_history('tool',gathered_information) 
    prompts.switch_system_prompt('reasoned_answer_prompt')

    print(gathered_information)

    final_response_stream = await tokenize_and_generate(max_new_tokens=512,temperature=settings['temperature'])

    print(final_response_stream)

    await cl.Message(content=final_response_stream.message.content).send()


    '''
    if response.message.tool_calls:
        for tool in response.message.tool_calls:
            print(f"Tool called: {tool.function.name}")
            print(f"Arguments: {tool.function.arguments}")

            function_name = tool.function.name
            function_args = tool.function.arguments

            if function_name in available_tools:
                tool_to_call = available_tools[function_name]
                tool_output = await tool_to_call(**function_args)
                response_stream = await respond_with_tool_output(user_message.content,tool_output)
                return response_stream'''

async def explore_knowledge_graph(strategy):
    # Input  : strategy = {node:allowed_edges}
    # Output : LLM-compatible text for the next step.

    gathered_info_dict = dict()
    for node,allowed_edges in strategy.items():
        if(du.get_node_id(node)):
            for neighbor in du.get_all_neighbors(node)['all']:
                gathered_info_dict[du.get_node_name(neighbor)] = [du.knowledge_graph.nodes[du.get_node_id(neighbor)]['metadata']['summary']]
            
            all_edges = du.get_all_edges_of_type(node,allowed_edges)
            for neighbor in all_edges.keys():
                gathered_info_dict[du.get_node_name(neighbor)] += [element[1]['text'] for element in all_edges[neighbor]]


    gathered_info_list = []
    for key in gathered_info_dict.keys():
        gathered_info_list += [f"* {text}\n" for text in gathered_info_dict[key]]
        gathered_info_list += ["\n"]
    gathered_information = "Gathered Information:\n"+" ".join(gathered_info_list)
    
    return gathered_information

async def check_context_sufficiency(user_message, context):

    args_dict = dict()
    args_dict['query'] = user_message
    args_dict['context'] = context

    history = prompts.prompt_parser("context_sufficiency_check_prompt",args_dict)

    response = await tokenize_and_generate(max_new_tokens=512,temperature=0.3,template=SufficiencyCheck)

    print(response.is_sufficient)
    print(response.relevant_information)
    print(response.information_needed)
    print(response.additional_questions)

    chat_history.pop(-1) # Remove the user prompt

    additional_context = ""
    if(response.is_sufficient<8):
        for mcq in response.additional_questions:
            print(mcq.question)
            print(mcq.options)
            all_options = mcq.options+['Skip','Other']
            actions = [cl.Action(name=f'option_{idx}',payload={'option':idx},label=option) for idx,option in enumerate(all_options)]
            mcq_message = cl.AskActionMessage(content=f"Question : {mcq.question}",actions=actions)
            response = await mcq_message.send()

            if(all_options[response['payload']['option']]=='Skip'):
                mcq_message.content = "You skipped this answer."
                await mcq_message.update()

            elif(all_options[response['payload']['option']]=='Other'):
                other_response_msg = cl.AskUserMessage(content="Please provide your answer!")
                response = await other_response_msg.send()

                mcq_message.content = f"Question : {mcq.question}\nAnswer : {response['output']}"
                await mcq_message.update()
                await other_response_msg.remove()
                additional_context += f"Question : {mcq.question}\nAnswer : {response['output']}\n\n"

            else:
                mcq_message.content = f"Question : {mcq.question}\nAnswer : {all_options[response['payload']['option']]}"
                await mcq_message.update()
                additional_context += f"Question : {mcq.question}\nAnswer : {all_options[response['payload']['option']]}\n\n"

            #Mahamun looks like an old man with a wrinkly face, hooked nose and purple eyes.

    if(len(additional_context)>0):

        additional_context_dict = dict()
        additional_context_dict['role'] = 'user'
        additional_context_dict['content'] = additional_context
        chat_history.append(additional_context_dict)
        cl.user_session.set("chat_history",chat_history)

        cl.user_session.set("additional_context_flag",True)
        cl.user_session.set("additional_context",additional_context)

async def tokenize_and_generate(max_new_tokens=256,temperature=0.6,template=None,to_stream=False,use_tools=False):

    chat_history = cl.user_session.get("chat_history")
    settings = cl.user_session.get("settings")

    '''
    print("-------------------------------------------------")
    for message in chat_history:
        print()
        print(f"{message['role']} :\n{message['content']}")
    print("-------------------------------------------------")
    '''


    if(use_tools):
        tools = tools_list
    else:
        tools = None

    if(template):
        #print('Schema : ',template.model_json_schema())
        llm_output = await ollama_client.chat(
                                    model=config['model_id'],
                                    messages=chat_history,
                                    options={
                                                'temperature': temperature,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                                'num_predict': max_new_tokens       # Equivalent to max tokens (maximum tokens to generate)
                                            },
                                    format=template.model_json_schema(),
                                    tools = tools
                                    )

        print(llm_output.message.content)
        try:
            response_stream = template.model_validate_json(llm_output.message.content)
        except:
            pass

        response_stream = repair_json(llm_output.message.content)
        response_stream = template.model_validate_json(response_stream)

    else:
        response_stream = await ollama_client.chat(
                                    model=config['model_id'],
                                    messages=chat_history,
                                    #tools=tools_json,
                                    options={
                                                'temperature': temperature,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                                'num_predict': max_new_tokens       # Equivalent to max tokens (maximum tokens to generate)
                                            },
                                    stream = to_stream,
                                    tools = tools
                                    )

    return response_stream

async def respond_with_tool_output(user_query,tool_output):

    chat_history = cl.user_session.get("chat_history")
    settings = cl.user_session.get('settings')

    chat_history.append({"role": "tool","content": tool_output})
    chat_history.append({"role": "user","content": user_query})

    cl.user_session.set("chat_history",chat_history)
                        
    # 5. Second API call: Send history back so the model can read the tool output and reply to user
    print("\nSending tool output back to the model for final response...")

    final_response_stream = await tokenize_and_generate(max_new_tokens=512,temperature=settings['temperature'],to_stream=True)

    return final_response_stream

async def get_rag_context(user_query,categories,strategy,thinking_type):

    # Resolve typos in LLM output
    #matches,user_query = await cl.make_async(du.check_for_similar_nodes)(user_query)

    clauses = sem.advanced_split_and_repair(user_query)
    context, nodes, edges = await cl.make_async(du.get_graph_rag_context)(user_query=clauses,strategy=strategy,categories=[label_map[categories].lower()])

    cl.user_session.set("relevant_nodes", nodes)
    cl.user_session.set("relevant_edges", edges)

    await check_context_sufficiency(user_query,context)

    return context

#############

# MAIN APP START

#############

tools_list = [tool_desc for tool_desc in tools.values()]
print(tools_list)
print()

available_tools = {
    "get_additional_context":get_rag_context
}

helpers = [ {"id":"Ideate", "icon":"lightbulb", "description":"Add ideas to knowledge base"},
            {"id":"Find", "icon":"search", "description":"Find a node or an edge in the graph."},
            {"id":"Analyze", "icon":"brain", "description":"Analyze uploaded text"},
            {"id":"Update", "icon":"list-restart", "description":"Update internal knowledge base"},
            {"id":"View", "icon":"eye", "description":"View a card"},
            {"id":"Wiki", "icon":"tag", "description":"Find in Wiki"},
            {"id":"Forget", "icon":"trash", "description":"Forget the Current Conversation"}]

du.fig_width = 170
du.fig_height = 80
@cl.on_settings_update
async def on_settings_update(settings: dict):

    cl.user_session.set("settings", settings)

    print(cl.user_session.get("settings"))

@cl.on_chat_start
async def on_chat_start():

    #global mlx_executor 

    props = dict()
    props['name'] = ""
    props['layer'] = ""
    props['type'] = ""
    props['desc'] = ""
    props['neighbors'] = []

    graph_element = cl.CustomElement(
                    name="GraphInfo",
                    display="side",
                    props= props
                )

    node_card_element = cl.CustomElement(
                    name="NodeCard",
                    display="inline",
                    props= props
                )

    info_props = {"text":""}

    cl.user_session.set("graph_element", graph_element)
    cl.user_session.set("node_card_elements_side", [])
    cl.user_session.set("node_card_element", node_card_element)
    cl.user_session.set("chat_history", [])

    cl.user_session.set("relevant_nodes", [])
    cl.user_session.set("relevant_edges", None)

    cl.user_session.set("additional_context", "")
    cl.user_session.set("additional_context_flag", False)
    cl.user_session.set("wiki_text","")

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

    # Define System prompt
    cl.user_session.set("chat_history", [])
    cl.user_session.set("entities_to_track", [])
    cl.user_session.set("awaiting_input_edge", False)
    cl.user_session.set("awaiting_input_node", False)
    cl.user_session.set("payload", None)

    await cl.context.emitter.set_commands(helpers)
    
    du.reload_graph()

    counters = dict()
    counters['fact'] = 0
    counters['topic'] = 0
    counters['lore'] = 0
    counters['plot'] = 0

    print(du.dataset)

    for node in du.knowledge_graph.nodes():
        print(f"{du.knowledge_graph.nodes[node]['text']} : {du.knowledge_graph.nodes[node]['updated']}")
        
    await cl.Message(content="Hello! I'm Quill, your novel-writing assistant! How can I help you today?").send()

    #for node in du.knowledge_graph.nodes():
    #    du.knowledge_graph.nodes[node]['metadata']['summary_updated'] = 0
    #await summarize_nodes()

@cl.on_message
async def on_message(user_message: cl.Message):

    selected_mode = user_message.command

    if(selected_mode=='Analyze'):
        #await reorder_plot(user_message.content)
        #await summarize_nodes()
        await show_information(user_message.content)
        #await cluster_lore()
    elif(selected_mode=='Update'):
        await summarize_nodes()
        await save_to_faiss()
    elif(selected_mode=='Ideate'):
        await extract_knowledge_graph(user_message.content,'lore_graph','idea')
    elif(selected_mode=='Wiki'):
        await find_in_wiki(user_message.content)
    elif(selected_mode=='View'):
        await show_knowledge_graph(user_message.content)
    elif(selected_mode=='Find'):
        await show_information(user_message.content)
    elif(selected_mode=='Forget'):
        cl.user_session.set("chat_history", [])
    else:
        
        if(True):

            #await decompose_text(user_message.content,['Krugrals (location)'])

            final_response_stream = await choose_and_use_tool(user_message)
            content = ""

            actions = [cl.Action(name='show_context_graph',payload={'nodes':cl.user_session.get("relevant_nodes"),'edges':cl.user_session.get("relevant_edges")},label='Show Context')]

            if(cl.user_session.get("additional_context_flag")):
                actions += [cl.Action(name='save_additional_context',payload={'context':cl.user_session.get("additional_context")},label='Save Additional Context')]

            msg = cl.Message(content="",actions=actions)
            for chunk in final_response_stream:
                print(chunk.message.content, end='', flush=True)
                await msg.stream_token(chunk.message.content)
                content += chunk.message.content

                if chunk.get('done', False):
                    prompt_tokens = chunk.get('prompt_eval_count',0)
                    output_tokens = chunk.get('eval_count',0)
                    print('Input prompt length is currently',prompt_tokens+output_tokens)

            cl.user_session.set("source_message", msg)
            response = await msg.update()


            chat_history = cl.user_session.get("chat_history")

            chat_history.append({
                                    "role": "assistant",
                                    "content": content,
                                })

            cl.user_session.set("chat_history",chat_history)















