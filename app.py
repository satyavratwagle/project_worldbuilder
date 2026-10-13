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
from utils.pydantic_schema import ReasonedResponse, DecomposedText, NodeSummary, TextTuple, PhaseList, Reordering, SufficiencyCheck, label_map
from utils.datastore_utils import DatastoreUtilities, slugify_key

from json_repair import repair_json
from pathlib import Path

import ollama
import copy

# remove later
import torch

with open('utils/tools.json', "r", encoding="utf-8") as f:
    tools = json.load(f)

with open('utils/prompts.json', "r", encoding="utf-8") as f:
    prompts_lookup = json.load(f)

helpers = [ {"id":"Ideate", "icon":"lightbulb", "description":"Add ideas to knowledge base"},
            {"id":"Find", "icon":"search", "description":"Find a node or an edge in the graph."},
            {"id":"Analyze", "icon":"brain", "description":"Analyze uploaded text"},
            {"id":"Update", "icon":"list-restart", "description":"Update internal knowledge base"},
            {"id":"View", "icon":"eye", "description":"View a card"},
            {"id":"Add Alias", "icon":"tag", "description":"Add Metadata"},
            {"id":"Forget", "icon":"trash", "description":"Forget the Current Conversation"}]

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
    wiki_schema = json.load(f)

named_entities = ['character','location','artifact','faction','event','definition']

#"Qwen/Qwen2.5-0.5B-Instruct" #
punctuation_tuple = tuple(string.punctuation)

# Engineering functions (synchronous)

def update_system_prompt(prompt_name):

    system_prompt = copy.deepcopy(prompts_lookup[prompt_name]['system'])
    system_prompt['content'] = '\n'.join(system_prompt['content'])

    chat_history = cl.user_session.get("chat_history")
    if(len(chat_history)>0):
        chat_history[0] = system_prompt
    else:
        chat_history = [system_prompt]
    cl.user_session.set("chat_history",chat_history)

def uppercase(text):
    return text[0].upper()+text[1:]

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

async def update_edit_element(text):

    info_edit_element = cl.user_session.get("info_edit_element")
    info_edit_element.props['text'] = text
    cl.user_session.set("info_edit_element", info_edit_element)
    await info_edit_element.update()

    return info_edit_element

@cl.cache
def load_modules():

    du = DatastoreUtilities(config,'worldbuilding')
    du.load_embedding_model()

    sem = SemanticTools(config)
    sem.load_extraction_model()
    #sem.load_nli_model()
    #sem.load_zsc_model()

    return du,sem

#du = DatastoreUtilities(config,'test','lore',True)
#du.load_embedding_model()
#sem = SemanticTools(config)
#sem.load_extraction_model()
du,sem = load_modules()

#############

# Chainlit UI Functions

#############

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
                chosen_selections[key] = (selection_response[key][1],selection_response[key][2])

    if(show_response):
        element_msg.content = f'I will find information about {', '.join([f"**{key}**" for key in chosen_selections.keys()])}!'
        await element_msg.update()

    return chosen_selections

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

#############

# Callbacks

#############
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

    edit_element = await update_edit_element(text)

    print(edit_element.props)
        
    edit_message = cl.Message(
                content="",
                elements=[edit_element]
            )

    response = await edit_message.send()

    print(response)

    #cl.user_session.set("awaiting_input_edge", True)
    #cl.user_session.set("payload", payload)
    #response = await cl.Message(content=f"Please modify '{text}'' in the chat.").send()

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

@cl.action_callback("show_context_graph")
async def on_show_context_graph(action: cl.Action):

    #msg: cl.Message = cl.user_session.get("source_message")

    nodes = action.payload.get("nodes")
    edges = action.payload.get("edges")
    
    fig = du.create_network_figure(nodes,target_edges=[tuple(e) for e in edges])
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")

    msg = await cl.Message(content="Hover over the nodes or edges for more information!",elements=[element]).send()
    #msg.elements.append(element)
    #await msg.update()

@cl.action_callback("save_additional_context")
async def on_save_additional_context(action: cl.Action):

    #msg: cl.Message = cl.user_session.get("source_message")

    context = action.payload.get("context")
    resolved_text, entity_dict = await resolve_coreferences(context)
    nodes_to_add = await decompose_text_loop(resolved_text, entity_dict)
    new_nodes, new_edges = await decomposition_post_processing(nodes_to_add)

@cl.action_callback("add_modify")
async def add_modify(action: cl.Action):

    modify_this = action.payload.get("modify")
    response = await cl.AskUserMessage(content="Please describe what you'd like to add or modify!").send()
    user_response = response['output']

    if(modify_this=='node'):

        node = du.get_relevant_node(user_response,threshold=0.8)
        if(node):
            # Show user existing node and ask if they want to modify or delete.
            # Modify for `plot` and `lore` will be updating the desc.
            # Modify for `topic` will be adding an alias (For now.)
            fig = du.create_network_figure([node],target_edges=[])
            element = cl.Plotly(name="network_graph", figure=fig, display="inline")
            actions = [cl.Action(name=name,payload={'selection':name.lower()},label=name) for name in ['Modify','Delete','Done']]
            msg = await cl.Message(content="Found Node!",elements=[element]).send()
            response = await cl.AskActionMessage(content="Do you want to keep this node?",actions=actions).send()

            if(response['payload']['selection']=='delete'):
                du.remove_node(node)
                du.save_graph()
            elif(response['payload']['selection']=='modify'):
                if(du.get_node_layer(node) in ['plot','lore']):
                    response = await cl.AskUserMessage(content="Please type in the modified description!").send()
                    nodes_to_add = [(response['output'],du.get_node_layer(node))]
                    print(nodes_to_add)
                    new_nodes, new_edges = await decomposition_post_processing(nodes_to_add)

        else:
            # Ask user if they want to create a node here.
            actions = [cl.Action(name=name,payload={'selection':name.lower()},label=name) for name in ['Yes','No']]
            response = await cl.AskActionMessage(content="I did not find any relevant nodes. Would you like to create one?",actions=actions).send()
            if(response['payload']['selection']=='yes'):
                resolved_text, entity_dict = await resolve_coreferences(user_response)
                nodes_to_add = await decompose_text_loop(resolved_text, entity_dict)
                new_nodes, new_edges = await decomposition_post_processing(nodes_to_add)

    elif(modify_this=='edge'):

        head,tail = du.get_relevant_edge(user_response,threshold=0.5)

        if(head and tail):
            # Show user existing edge and ask if they want to delete. (Modify is immaterial here.)
            fig = du.create_network_figure([head,tail],target_edges=[(head,tail)])
            element = cl.Plotly(name="network_graph", figure=fig, display="inline")
            actions = [cl.Action(name=name,payload={'selection':name.lower()},label=name) for name in ['Delete','Done']]
            msg = await cl.Message(content="Found Connection!",elements=[element]).send()
            response = await cl.AskActionMessage(content="Do you want to keep this connection?",actions=actions).send()

            if(response['payload']['selection']=='delete'):
                du.remove_edge(head,tail)
                du.save_graph()
        else:
            # Cannot manually "create" and edge.
            await cl.Message(content="Did not find any relevant connections!").send()
            

    await cl.context.emitter.task_end()


    
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

async def retrieve_graph_rag(query,threshold=0.4,k=10,hops=1):

    context,context_edges = du.get_graph_rag_context(query,threshold,k)
    print('retrieve_graph_rag')
    print(context)
    print()

    return context,context_edges

#############

# Composite Utilities

#############

async def resolve_coreferences(user_text):

    user_text,_ = du.resolve_aliases(user_text)
    existing_alias_dict = du.get_alias_dict(user_text)

    print(f"\nInitial Resolution : {user_text}")
    print(f"\nExisting Entity Dict : {list(existing_alias_dict.keys())}")

    doc,entity_coreferences = sem.get_alias_clusters(user_text,existing_alias_dict)
    entity_dict,replacements = sem.match_clusters_against_aliases(doc,entity_coreferences,existing_alias_dict)
    resolved_text = sem.resolve_aliases(doc,replacements)

    return resolved_text, entity_dict    

async def choose_entities_to_track(resolved_text, entity_dict):

    # Get user to add More Entities to the nodes to extract
    extracted_entities = sem.named_entity_extraction(resolved_text,labels=['Character','Location','Artifact','Faction','Event'])
    selection_response = await show_checklist(extracted_entities,message='Please select the topics to add to the story graph!')
    selected_keys = set(selection_response.keys())

    # Remove keys from entity dict based on chosen entities
    keys_to_remove = []
    for key in entity_dict.keys():
        if(not(key in selected_keys)):
            keys_to_remove.append(key)

    for key in keys_to_remove:
        entity_dict.pop(key)

    # Add keys from entity dict based on selections
    keys_to_add = []
    for key in selected_keys:
        if(not(key in entity_dict.keys())):
            keys_to_add.append(key)

    for key in keys_to_add:
        entity_dict[key] = dict()
        entity_dict[key]['type'] = selection_response[key][0]
        entity_dict[key]['aliases'] = [key]

    print(f"\nFinal Entity Dict : {list(entity_dict.keys())}")
    
    # Check later if this is needed.


    alias_message = "I found aliases to some of the topics as follows!"
    entities_with_aliases = []
    for entity in entity_dict.keys():
        if(len(entity_dict[entity]['aliases'])>0):
            alias_message += f"\n{entity} is referred to as {", ".join(entity_dict[entity]['aliases'])}"
    alias_message += "\nIs that okay?"

    actions = [cl.Action(name=name,payload={'alias':name},label=name) for name in ['Yes','No']]
    alias_check_msg = cl.AskActionMessage(content=alias_message,actions=actions,timeout=30)

    response = await alias_check_msg.send()
    if(response['name']=='Yes'):
        await alias_check_msg.remove()
    else:
        alias_check_msg.content = f"I am unsure of the names of {", ".join([entity for entity in entities_with_aliases])}!\nPlease add your information again!"
        await alias_check_msg.update()
        return None

    return entity_dict

async def decompose_text_loop(resolved_text,entity_dict):

    # Combine all names
    alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias,type in entity_dict.items())})\b"
    alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

    nodes_to_add = []
    for chunk in resolved_text.split('\n\n'):

        for e_idx,selected_entity in enumerate(entity_dict.keys()):
            # Filter Chunk to only consider sentences where the entity is mentioned
            
            chunk_words = chunk.split(" ")
            chunk_starter = " ".join(chunk_words[:min(len(chunk_words),4)])

            async with cl.Step(name=f"{chunk_starter}... to find information about {selected_entity}",default_open=True) as step:
                formatted_chunk = "\n".join(sem.split_into_sentences(chunk))
                decomposed_tuples = await decompose_text(formatted_chunk,topics=[selected_entity])

                print('\nRaw Output : ',decomposed_tuples)

                # Change this : Tuples will have labels as well.
                tuples_to_add = await show_edges_to_add(decomposed_tuples,message=f"Please select information about {selected_entity} to remember!")

                for tuplet in tuples_to_add:
                    text,type = tuplet
                    nodes_to_add.append((text,type))

    return nodes_to_add

async def decomposition_post_processing(nodes_to_add):

    
    node_indices_to_keep = du.resolve_text([text for text,type in nodes_to_add])
    nodes_to_add = [nodes_to_add[idx] for idx in node_indices_to_keep]

    # Check against existing nodes for similarity and remove redundant nodes here.
    to_compare = [text for text,type in nodes_to_add]
    comparison_dict = du.compare_against_existing_nodes(to_compare)

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

                response = await cl.AskActionMessage(content=f"Found a node similar to **{to_compare[idx]}**! Which one shuold I keep?",actions=actions).send()
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
    print(f"\nNodes to Add : {list(set(nodes_to_add))}")
    for text,type in list(set(nodes_to_add)):
        new_nodes.append(du.add_node(text,type,"misc"))

    new_edges = []
    # topic --> lore
    new_edges += du.extrapolate_descriptions()
    # topic --> plot
    new_edges += du.extrapolate_involvements()
    # topic --> topic
    new_edges += du.extrapolate_relationships()

    new_edges = [(head,tail) for head,tail,edge in new_edges]

    du.save_graph()

    return new_nodes,new_edges

#############

# Knowledge Graph Tools

#############

async def extract_knowledge_graph(og_text,graph_name='graph',graph_type='story'):

    if(graph_type=='idea'):
        graph_name = 'lore_graph'

    async with cl.Step(name="Topic Selection",default_open=True) as step:
        # Extract (Named Entities from Text + Nodes from Graph + Additional User Specified Entities)

        resolved_text, entity_dict = await resolve_coreferences(og_text)
        entity_dict = await choose_entities_to_track(resolved_text, entity_dict)

    # Final entity dict should only contain the names of the entities to be added.
    # Final resolved text should be resolved against all aliases, irrespective of whether or not they have been added.
        
    # Extract node names from the existing Graph

    # Add all the new "topic" nodes to the graph and reload the maps
    for entity in entity_dict.keys():
        node_id = du.add_node(entity,"topic",entity_dict[entity]['type'],aliases=entity_dict[entity]['aliases'])
    du.reload_lookups()

    # Summarize the scenes
    
    nodes_to_add = await decompose_text_loop(resolved_text, entity_dict)

    # Remove duplicate nodes.
    new_nodes, new_edges = await decomposition_post_processing(nodes_to_add)

    fig = du.create_network_figure(new_nodes,target_edges=new_edges)
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    msg = await cl.Message(content="Updated the knowledge graph!",elements=[element]).send()
            
async def audit_graph(graph_name='graph'):

    # Analyze the Graph and check if any nodes are too similar.

    to_compare = [du.get_node_name(node) for node in du.knowledge_graph.nodes()]
    comparison_dict = du.compare_against_existing_nodes(to_compare,threshold=0.85)

    for idx,similar_nodes in comparison_dict.items():
        if(len(similar_nodes)>0):
            for node in similar_nodes:
                print(f"{to_compare[idx]} --> {du.get_node_name(node)}")

        else:
            print(f"{to_compare[idx]} is good to go!")

async def summarize_nodes():

    updated_nodes = []
    for node in du.get_all_nodes():

        # Check if the node needs to be updated
        if(du.has_new_edges(node['id'])):

            #du.reset_node_summary(node['id'])
            edge_info_clusters = du.cluster_edge_info(node['id'])
            print(edge_info_clusters)

            message = cl.Message(content=f"## {node['name']}")
            await message.send()

            history = []
            props_dict = dict()
            props_dict["node_name"] = node['name']

            for key in [label_map[schema] for schema in TextTuple.model_json_schema()['properties']['label']['enum']]:
                if(key in edge_info_clusters.keys()):
                    props_dict[key]  = f"{edge_info_clusters[key]}"
                else:
                    props_dict[key]  = f"No {key}-related information available."
            
            print(props_dict)
            print(f"\n***\n")

            history = prompts.get_node_summary_prompt(props_dict,history)
            cl.user_session.set("chat_history",history)

            print(history[-1]['content'])

            response = await tokenize_and_generate(temperature=0.3,max_new_tokens=1024, template=NodeSummary)


            message.content = f"## {node['name']}\n\n### Lore\n{response.lore}\n\n\n### Events\n{response.plot}\n"
            await message.update()

            du.add_to_node_summary(node['id'],'lore',response.lore,)
            du.add_to_node_summary(node['id'],'plot',response.plot,)

            updated_nodes.append(node['id'])
            #du.add_to_node_summary(node['id'],'attribute',response.attribute,)
            #du.add_to_node_summary(node['id'],'time_period',response.time_period,)

    du.save_graph()

    fig = du.create_network_figure(updated_nodes,target_edges=[])
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    msg = await cl.Message(content="Updated the highlighted nodes!",elements=[element]).send()

async def save_to_faiss():

    not_in_graph,not_in_dataset = du.calculate_graph_datastore_diff(write_to_kg2ds=False)

    graph_diff = len(not_in_graph)>0
    dataset_diff = not(all([du.get_node_layer(node)=='topic' for node in not_in_dataset]))

    if(graph_diff or dataset_diff):
        await cl.Message("Would you like to save this session?",actions=[cl.Action(name='update_faiss', payload={"label": "None"}, label="Save")]).send()
    else:
        await cl.Message("No nodes have been updated!").send()

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

async def add_node_alias(node_name):
    if(du.does_node_exist(node_name)):

        response = await cl.AskUserMessage(content=f"Please provide an alias for {node_name}!", timeout=30).send()
        alias = response['output'].strip()
        aliases = du.add_alias(node_name,alias)
        await cl.Message(content=f"Aliases for {node_name} are {", ".join(aliases)}").send()
    else:
        await cl.Message(content=f"No topic named '{node_name}'' found!").send()

#############

# Core LLM Functions

#############

async def show_knowledge_graph(user_message):

    if(user_message in ['plot','lore','topic']):
        layer = user_message
        nodes = [node_id for node_id,desc in du.get_nodes_in_layer(layer)]
        edges = du.get_edges_in_layer(layer)
    else:
        node = du.get_relevant_node(user_message,threshold=0.8)
        if(node):
            nodes,edges = du.get_node_stack(node)
            print(nodes)
            layer = 'topic'
        else:
            actions = [cl.Action(name=uppercase(name),payload={'layer':name.lower()},label=uppercase(name)) for name in ['plot','lore','topic','none']]
            response = await cl.AskActionMessage("Please select the knowledge graph layer you want to view!",actions=actions).send()
            if(not(response['payload']['layer']=='none')):
                layer = response['payload']['layer']
                nodes = [node_id for node_id,desc in du.get_nodes_in_layer(layer)]
                edges = du.get_edges_in_layer(layer)

    print(nodes)
    actions = [cl.Action(name='add_modify',payload={'modify':'node'},label='Add/Modify Node'),cl.Action(name='add_modify',payload={'modify':'edge'},label='Add/Modify Connection')]
    fig = du.create_network_figure(nodes,target_edges=edges,layer=layer)
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    msg = await cl.Message(content="Hover over the nodes or edges for more information!",elements=[element],actions=actions).send()


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
        update_system_prompt('plot_reordering_prompt')
        chat_history = cl.user_session.get("chat_history")

        args_dict = dict()
        args_dict['phases'] = context
        args_dict['outline'] = resolved_text

        history = prompts.get_reordering_prompt(args_dict,[])
        chat_history.append(history[-1])
        cl.user_session.set("chat_history",chat_history)

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

async def decompose_text(text,topics=[],temperature=0.2):
    # topics : list (List of topics to focus on)

    # Clear out chat history to avoid cross-contamination
    chat_history = cl.user_session.get("chat_history")
    props_dict = dict()
    props_dict['topics']                = '\n'.join([f"{topic}" for idx,topic in enumerate(topics)])
    props_dict['text']                  = text
    chat_history = prompts.get_text_decomposition_prompt(props_dict,history=[])
    cl.user_session.set("chat_history",chat_history)
    update_system_prompt("text_decomposition_prompt")

    settings = cl.user_session.get('settings')

    llm_response = await tokenize_and_generate(temperature=0.15,max_new_tokens=1024,template=DecomposedText)

    decomposed_tuples = []
    for tuplet in llm_response.decomposed_text:
        decomposed_tuples.append([tuplet.atomic_sentence,label_map[tuplet.label],tuplet.topic])

    return decomposed_tuples

async def retrieve_context(topic,entity_threshold=0.25,rag_threshold=0.5,k=10):

    settings = cl.user_session.get('settings')
    #topic_with_context = await add_entity_context(topic,entity_threshold)
    context_text,context_edges = await retrieve_graph_rag(topic,threshold=settings['rag_threshold'],k=k)

    return context_text,context_edges

async def check_context_sufficiency(user_message, context):

    update_system_prompt('context_sufficiency_check_prompt')
    chat_history = cl.user_session.get("chat_history")

    args_dict = dict()
    args_dict['query'] = user_message
    args_dict['context'] = context

    history = prompts.get_context_sufficiency_check_prompt(args_dict,[])
    chat_history.append(history[-1])
    cl.user_session.set("chat_history",chat_history)

    print(chat_history[-1]['content'])

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

    for message in chat_history:
        print()
        print(f"{message['role']} :\n{message['content']}")

    if(use_tools):
        tools = tools_list
    else:
        tools = None

    if(template):
        response_stream = ollama.chat(
                                    model=config['model_id'],
                                    messages=chat_history,
                                    options={
                                                'temperature': temperature,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                                'num_predict': max_new_tokens       # Equivalent to max tokens (maximum tokens to generate)
                                            },
                                    format=template.model_json_schema(),
                                    tools = tools
                                    )
        response_stream = template.model_validate_json(response_stream.message.content)
    else:
        response_stream = ollama.chat(
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

async def choose_and_use_tool(user_message):

    settings = cl.user_session.get("settings")

    update_system_prompt("tool_use_prompt")
    chat_history = cl.user_session.get("chat_history")

    # Check for typos

    matches,resolved_text = await cl.make_async(du.check_for_similar_nodes)(user_message.content)

    for source,replacement in matches:
        print(f"Replaced : {source} ---> {replacement}")

    print("Resolved : ",resolved_text)

    chat_history.append({
                            'role': 'user', 
                            'content': f"{resolved_text}"
                        })

    cl.user_session.set("chat_history",chat_history)

    response = await tokenize_and_generate(max_new_tokens=512,temperature=0.3,use_tools=True)

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
                return response_stream

async def respond_with_tool_output(user_query,tool_output):

    chat_history = cl.user_session.get("chat_history")
    settings = cl.user_session.get('settings')

    chat_history.append({
                            "role": "tool",
                            "content": tool_output,
                        })
    chat_history.append({
                            "role": "user",
                            "content": user_query,
                        })

    update_system_prompt('reasoned_answer_prompt')
    cl.user_session.set("chat_history",chat_history)
                        
    # 5. Second API call: Send history back so the model can read the tool output and reply to user
    print("\nSending tool output back to the model for final response...")

    final_response_stream = await tokenize_and_generate(max_new_tokens=512,temperature=settings['temperature'],to_stream=True)

    return final_response_stream

async def default_tool(user_message):
    context_text,context_edges = await retrieve_context(user_message)
    update_system_prompt('reasoned_answer_prompt')
    response_stream = await respond_with_tool_output(context_text)
    return response_stream

async def get_rag_context(user_query,categories,strategy,thinking_type):

    # Resolve typos in LLM output
    #matches,user_query = await cl.make_async(du.check_for_similar_nodes)(user_query)

    clauses = sem.advanced_split_and_repair(user_query)

    print(f"Query sent to RAG : {clauses}")

    context, nodes, edges = await cl.make_async(du.get_graph_rag_context)(user_query=clauses,strategy=strategy,categories=[label_map[categories].lower()])

    if(thinking_type=='answer'):
        update_system_prompt('reasoned_answer_prompt')
    elif(thinking_type=='brainstorm'):
        update_system_prompt('brainstorming_prompt')

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

@cl.on_settings_update
async def on_settings_update(settings: dict):

    cl.user_session.set("settings", settings)

    print(cl.user_session.get("settings"))

@cl.on_chat_start
async def on_chat_start():

    #global mlx_executor 

    props = {"title":"",
                "subtitle":"",
                "edges":""}

    graph_element = cl.CustomElement(
                    name="GraphInfo",
                    display="side",
                    props= props
                )

    info_props = {"text":""}

    info_edit_element = cl.CustomElement(
                    name="EditEdge",
                    display="inline",
                    props= info_props
                )

    cl.user_session.set("graph_element", graph_element)
    cl.user_session.set("info_edit_element", info_edit_element)
    cl.user_session.set("chat_history", [])

    cl.user_session.set("relevant_nodes", [])
    cl.user_session.set("relevant_edges", None)

    cl.user_session.set("additional_context", "")
    cl.user_session.set("additional_context_flag", False)


    
    #mx.metal.clear_cache()
    #mlx_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    #loop = asyncio.get_running_loop()
    #await loop.run_in_executor(mlx_executor, _sync_load_models)

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
    
    #for id,node in du.knowledge_graph.nodes(data=True):
    #    print(f"Found Node {node['desc']} ({node['layer']})")
    #print(du.alias_pattern)

    '''
    du.knowledge_graph.clear_edges()
    output = du.extrapolate_descriptions()
    output = du.extrapolate_descriptions()
    output = du.extrapolate_relationships()
    du.save_graph()
    '''

    
    await cl.Message(content="Hello! I'm Quill, your novel-writing assistant! How can I help you today?").send()

@cl.on_message
async def on_message(user_message: cl.Message):

    selected_mode = user_message.command

    if(selected_mode=='Analyze'):
        await reorder_plot(user_message.content)
    elif(selected_mode=='Update'):
        await save_to_faiss()
    elif(selected_mode=='Ideate'):
        await extract_knowledge_graph(user_message.content,'lore_graph','idea')
    elif(selected_mode=='Add Alias'):
        await add_node_alias(user_message.content)
    elif(selected_mode=='View'):
        await show_knowledge_graph(user_message.content)
    elif(selected_mode=='Find'):
        await find(user_message.content)
    elif(selected_mode=='Forget'):
        cl.user_session.set("chat_history", [])
    else:
        # Make this cleaner.
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

            #result = await elaborate(user_message.content)
            #print(result)

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






