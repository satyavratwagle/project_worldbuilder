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
from utils.pydantic_schema import ReasonedResponse, DecomposedText, NodeSummary, TextTuple, PhaseList, Reordering, label_map
from utils.datastore_utils import DatastoreUtilities, slugify_key

from json_repair import repair_json
from pathlib import Path

import ollama
import copy

with open('utils/tools.json', "r", encoding="utf-8") as f:
    tools = json.load(f)

with open('utils/prompts.json', "r", encoding="utf-8") as f:
    prompts_lookup = json.load(f)

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

    du = DatastoreUtilities(config,'demo','lore',True)
    du.load_embedding_model()

    sem = SemanticTools(config)
    sem.load_extraction_model()
    #sem.load_nli_model()
    #sem.load_zsc_model()

    return du, sem

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
        items_list.append({"id":idx,"text":tuplet[0],"label":tuplet[1],"topics":tuplet[2]})

    
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
                edges_to_add.append([selection_response[key][0],selection_response[key][3],selection_response[key][4]])

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


#############

# Datastore Operations

#############

@cl.action_callback("update_faiss")
async def update_datastore(action: cl.Action):

    msg = cl.Message(content="Updating Knowledge Base...")

    await msg.send()

    dataset,index = await make_async(du.filter_dataset)()
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

# Knowledge Graph Tools

#############

async def extract_knowledge_graph(og_text,graph_name='graph',graph_type='story'):

    if(graph_type=='idea'):
        graph_name = 'lore_graph'

    async with cl.Step(name="Topic Selection",default_open=True) as step:
        # Extract (Named Entities from Text + Nodes from Graph + Additional User Specified Entities)

        og_text,_ = du.resolve_aliases(og_text)
        print("\nInitial Alias Resolution : \n",og_text)
        extracted_entities = sem.named_entity_extraction(og_text,labels=['Character','Location','Artifact','Faction','Event'])

        # Get user to add More Entities to the nodes to extract
        selection_response = await show_checklist(extracted_entities,message='Please select the topics to add to the story graph!')
        selected_keys = set(selection_response.keys())
        print('\nSelection Response : ',selection_response)

        entity_dict = dict()
        for entity,type in extracted_entities.items():

            entity_dict[entity] = dict()
            entity_dict[entity]['aliases'] = []

            if(entity in selection_response.keys()):

                # Entities extracted by NER and selected to add

                entity_dict[entity]['type'] = selection_response[entity][0]
                entity_dict[entity]['to_add'] = True
            else:

                # Entities extracted by NER but not selected to add

                entity_dict[entity]['to_add'] = False

        for entity in selected_keys:
            if(not(entity in extracted_entities.keys())):

                # Entities added by the user, but not found by NER

                entity_dict[entity] = dict()
                entity_dict[entity]['type'] = selection_response[entity][0]
                entity_dict[entity]['aliases'] = []

        # Replace the known aliases in the chunk with node names
        similar_pairs = du.check_nodes_for_replacement(entity_dict.keys())  

        print('\nNodes for Replacement : ',similar_pairs)   

        # Get user response on what to replace
        print("\nEntity Dict Before Replacement : ",entity_dict)
        replacements = []
        for key,node in similar_pairs:

            if(key in selection_response.keys()):
                actions = [cl.Action(name=name,payload={'alias':name},label=name) for name in ['Yes','No']]
                alias_confirmation_msg = cl.AskActionMessage(content=f"'{key}' is similar to an existing topic called '{node['name']}'. Are these the same?",actions=actions,timeout=30)
                response = await alias_confirmation_msg.send()
                if(response['name']=='Yes'):
                    replacements.append((key,node))
                    selected_keys.discard(key)
                    selected_keys.add(node['name'])
                await alias_confirmation_msg.remove()
            else:
                replacements.append((key,node))

        for key,node in replacements:
            og_text = og_text.replace(key,node['name'])

            entity_dict[node['name']] = entity_dict.pop(key)
            entity_dict[node['name']]['type'] = node['type']
            entity_dict[node['name']]['aliases'] = []
            entity_dict[node['name']]['to_add'] = True

        print("\nEntity Dict After Replacement : ",entity_dict)

        keys_to_remove = []
        for key in entity_dict.keys():
            if(not(key in selected_keys)):
                keys_to_remove.append(key)

        for key in keys_to_remove:
            entity_dict.pop(key)

        print("\nEntity Dict After Removal of unselected keys : ",entity_dict)

        # Add entities from the text that were not selected, but are nodes
        existing_entities = list([node for node in set(du.node_finder.findall(og_text)) if len(node.strip())>0])

        for entity in existing_entities:
            if(not(entity) in entity_dict.keys()):
                entity_dict[entity] = dict()
                entity_dict[entity]['type'] = du.knowledge_graph.nodes[du.alias_map[entity]]['type']
                entity_dict[entity]['aliases'] = du.knowledge_graph.nodes[du.alias_map[entity]]['aliases']
                entity_dict[entity]['to_add'] = False

        print("\nEntity Dict After Adding existing nodes : ",entity_dict)

        print('\nResolving Corefs')
        doc,entity_coreferences = sem.get_alias_clusters(og_text,entity_dict)

        alias_message = "I found aliases to some of the topics as follows!"
        entities_with_aliases = []
        for entity_tuple in entity_coreferences:
            if(len(entity_tuple[2])>0):
                alias_message += f"\n{entity_tuple[1]} is referred to as {", ".join(entity_tuple[2])}"
                entities_with_aliases.append(entity_tuple[1])
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

        # Update aliases with those found in the text
        for entity_tuple in entity_coreferences:
            if(entity_tuple[1] in entity_dict.keys()):
                entity_dict[entity_tuple[1]]['aliases'] = entity_tuple[2]

        resolved_text = sem.resolve_coreferences(doc,entity_coreferences)
        print('\nCoreference Resolved Text : ',resolved_text)
        print('\nFinal Entity Dict :',entity_dict)
        
    
    # Extract node names from the existing Graph

    # Combine all names
    alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias,type in entity_dict.items())})\b"
    alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

    # Summarize the scenes
    chunk_idx = 1
    nodes_to_add = []
    edges_to_add = []
    for chunk in resolved_text.split('\n\n'):

        entities = alias_finder.findall(chunk)
        entities = list(set([e for e in entities if entity_dict[e]['to_add']]))
        entities_with_types = [f"{e}" for e in entities]
        print(entities_with_types)

        # Change the type of the entity if needed.
        #scene_entities = [e['text'] for e in entities[0] if e['text'] in aliased_entity_names]+user_defined_entity_names

        for e_idx,selected_entity in enumerate(entities_with_types):
            # Filter Chunk to only consider sentences where the entity is mentioned

            
            chunk_words = chunk.split(" ")
            chunk_starter = " ".join(chunk_words[:min(len(chunk_words),5)])
            entity_name = selected_entity.split(' (')[0]

            async with cl.Step(name=f"{chunk_starter}... to find information about {entity_name}",default_open=True) as step:
                chunk_sentences = chunk.split('.')
                filtered_sentences = list(filter(lambda sentence:entity_name in sentence,chunk_sentences))

                decomposed_tuples = await decompose_text(chunk,topics=[selected_entity])

                print('\nRaw Output : ',decomposed_tuples)
                
                # Remove sentences that do not mention the topics of interest.
                filtered_tuples = [[tuplet[0],tuplet[1],alias_finder.findall(tuplet[0])] for tuplet in decomposed_tuples if len(alias_finder.findall(tuplet[0]))>0]
                print("\nFiltered Output : ",filtered_tuples)

                for idx,tuplet in enumerate(filtered_tuples):
                    head,tail = sem.split_sentence(tuplet[0])

                    to_add = dict()
                    to_add['heads'] = list(filter(lambda node: (len(node.strip())>0 and entity_dict[node]['to_add']),(list(set(alias_finder.findall(head)))+list(set(du.node_finder.findall(head))))))
                    to_add['tails'] = list(filter(lambda node: (len(node.strip())>0 and entity_dict[node]['to_add']),(list(set(alias_finder.findall(tail)))+list(set(du.node_finder.findall(tail))))))

                    filtered_tuples[idx][2] = to_add

                # May not need this.
                #cosine_sim = du.get_cosine_similarity(filtered_sentences)
                #filtered_sentences,_ = du.filter_similar_text(filtered_sentences,cosine_sim)

                # Convert Sentences to Edges
                added_edges = []
                # Change this : Tuples will have labels as well.
                tuples_to_add = await show_edges_to_add(filtered_tuples,message=f"Please select information about {entities[e_idx]} to remember!")

                #relationship_tuples = [(sentence,[topic]) for sentence in sentences_to_add for topic in alias_finder.findall(sentence)]
                #print(relationship_tuples)
                #sentences_to_add,topics = zip(*relationship_tuples)
                #triplets = sem.relationship_extraction(sentences_to_add,topics)

                for tuplet in tuples_to_add:
                    text,label,topics = tuplet

                    nodes_to_add += topics['heads']+topics['tails']

                    if(len(topics['heads'])>0 and len(topics['tails'])>0):
                        for head in topics['heads']:
                            for tail in topics['tails']:
                                edges_to_add.append((head,tail,text,label))
                    else:
                        for node in list(set(topics['heads']+topics['tails'])):
                            edges_to_add.append((node,node,text,label))

            new_nodes = []
            print(f"\nNodes to Add : {list(set(nodes_to_add))}")
            for node in list(set(nodes_to_add)):
                print(f"\n Entity Dict for {node} is {entity_dict[node]}")
                node_id = du.add_node(node,entity_dict[node]['type'],aliases=entity_dict[node]['aliases'])
                print(f"Added Node : {du.knowledge_graph.nodes[slugify_key(node)]}")
                new_nodes.append(node_id)

    # Remove redundant edges here.

    edge_idxs_to_keep = du.resolve_text([edge[2] for edge in edges_to_add])
    filtered_edges = [edges_to_add[idx] for idx in edge_idxs_to_keep]

    new_edges = []
    for edge in filtered_edges:
        head,tail,text,label = edge
        key = du.add_edge(head,tail,text,edge_label=label)
        new_edges.append((head,tail,key))

    # Find new connections between existing edges.
    connected_edges = du.find_new_edges()

    new_edges += connected_edges
    new_edges = list(set(new_edges))
    
    du.save_graph()

    fig = du.create_network_figure(new_nodes,target_edges=[tuple(e) for e in new_edges])
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")
    msg = await cl.Message(content="Updated the knowledge graph!",elements=[element]).send()
            
async def audit_graph(graph_name='graph'):

    # Identify new connections and resolve old ones.

    async with cl.Step(name="Finding New Connections",icon="lightbulb") as step:
        # Find similar edges and keep one.
        edges_to_resolve = du.resolve_edges()

        #du.save_graph()
        for similar_edges in edges_to_resolve:
            if(len(similar_edges)>1):
                actions=[cl.Action(name=edge[3], payload={"edge_data":edge}, label=f"{edge[3]}") for edge in similar_edges]
                response = await cl.AskActionMessage(content="Which description would you like to keep?",actions=actions,timeout=60).send()
                selected_edge = tuple(response['payload']['edge_data'])
                edges_to_delete = [edge for edge in similar_edges if not(edge==selected_edge)]
                du.knowledge_graph.remove_edges_from(edges_to_delete)

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

    if(len(du.get_nodes_to_add_to_faiss())>0):
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

#############

# Core LLM Functions

#############

async def reorder_plot(user_query):

    user_query,entity_dict = du.resolve_aliases(user_query)
    print(entity_dict)
    doc,entity_coreferences = sem.get_alias_clusters(user_query,entity_dict)
    print(entity_coreferences)
    user_query = sem.resolve_coreferences(doc,entity_coreferences)

    print(user_query)

    outline, text_to_reorder, reference_dict = du.get_plot_to_reorder(user_query,plot_similarity_threshold=0.8)

    if(len(text_to_reorder.strip())>0):
        args_dict = dict()
        args_dict['phases'] = ""
        args_dict['outline'] = outline
        args_dict['text'] = text_to_reorder

        history = prompts.get_reordering_prompt(args_dict,[])
        print(history[-1]['content'])

        cl.user_session.set("chat_history",history)

        reordering = await tokenize_and_generate(max_new_tokens=1024,temperature=0.0,template=Reordering)

        print(reordering)

        print(f"Thinking : {reordering.thinking}")
        if(reordering.sequential_events):
            for id in reordering.before:
                head,tail,key,text = reference_dict[id]
                print(f'Before :{text}')
            for id in reordering.after:
                head,tail,key,text = reference_dict[id]
                print(f'After :{text}')
        else:
            for key in reference_dict.keys():
                head,tail,key,text = reference_dict[key]
                print(f'Concurrent :{text}')



        '''for phase in phase_list.phases:
            current_phase = int(phase.index)
            for id in phase.ids:
                head,tail,key,text = reference_dict[id]
                du.set_plot_edge_phase(head,tail,key,current_phase)
                print(f"Assigned '{text}' to phase {current_phase}")
            print()'''

        #du.save_graph()
    else:
        await cl.Message(content="No relevant story information to reorder found!").send()

    return True

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

#############

# TOOLS LIST

#############

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

async def hypothesize(user_idea):

    # Switch out system prompt
    update_system_prompt('brainstorming_prompt')

    nodes = du.get_relevant_nodes(user_idea,k=5, threshold=0.4)

    with cl.Step(name=f"Hypothesizing about {user_idea}",icon="lightbulb") as step:

        neighbors = []
        for node in nodes:
            # Use all neighbors of a node
            neighbors += du.graph_multihop(node,1)

        neighborhood_context = du.get_context_from_neighborhood(neighbors)

        props_dict = dict()
        props_dict['topic'] = user_idea
        props_dict['local_context'] = neighborhood_context
        full_prompt = prompts.get_brainstorming_prompt(props_dict,[])
        response_stream = await respond_with_tool_output(full_prompt[-1]['content'])
        return response_stream

async def elaborate(topic_description):

    head = du.get_relevant_nodes(topic_description,k=1, threshold=0.4)[0]

    with cl.Step(name=f"Elaborating upon {head}",icon="lightbulb") as step:
        nodes = du.graph_multihop(head)
        paths = du.find_all_unique_paths_from(head,nodes)
        context = du.extract_path_info(paths)
        context_text = du.format_path_context(context)
        response_stream = await respond_with_tool_output(context_text)
        return response_stream

async def get_rag_context(user_query,categories,strategy,thinking_type):

    # Resolve typos in LLM output
    matches,user_query = await cl.make_async(du.check_for_similar_nodes)(user_query)

    print(f"Query sent to RAG : {user_query}")

    context, nodes, edges = await cl.make_async(du.get_graph_rag_context)(user_query=user_query,strategy=strategy,categories=[label_map[categories].lower()])

    if(thinking_type=='answer'):
        update_system_prompt('reasoned_answer_prompt')
    elif(thinking_type=='brainstorm'):
        update_system_prompt('brainstorming_prompt')

    cl.user_session.set("relevant_nodes", nodes)
    cl.user_session.set("relevant_edges", edges)

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
    
    for id,node in du.knowledge_graph.nodes(data=True):
        print(f"Found Node {node['name']} ({node['type']}), also known as {node['aliases']}")

    for edge in du.knowledge_graph.edges(keys=True,data=True):
        head,tail,key,text = edge
        du.set_plot_edge_phase(head,tail,key,0)
        print(du.knowledge_graph.edges[head,tail,key])

    du.all_aliases = [alias for node in du.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']]
    du.alias_map = {alias:node[0] for node in du.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']}
    alias_pattern = rf"\b({'|'.join([re.escape(alias) for alias in du.all_aliases])})\b"
    du.node_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

    await cl.Message(content="Hello! I'm Quill, your novel-writing assistant! How can I help you today?").send()

@cl.on_message
async def on_message(user_message: cl.Message):

    selected_mode = user_message.command

    if(selected_mode=='Analyze'):
        await reorder_plot(user_message.content)
    elif(selected_mode=='Update'):

        await audit_graph('lore_graph')
        await summarize_nodes()
        await save_to_faiss()

    elif(selected_mode=='Ideate'):
        await extract_knowledge_graph(user_message.content,'lore_graph','idea')

    elif(selected_mode=='Add Chapter'):
        # Add a list of stories already in the user list
        story_name = 'test'
        await extract_knowledge_graph(user_message.content,story_name,'story')
    elif(selected_mode=='View'):
        await show_node_info(user_message.content)
    elif(selected_mode=='Brainstorm'):
        await hypothesize(user_message.content)
    elif(selected_mode=='Metadata'):
        await update_metadata(user_message)
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

                msg = cl.Message(content="",actions=[cl.Action(name='show_context_graph',payload={'nodes':cl.user_session.get("relevant_nodes"),'edges':cl.user_session.get("relevant_edges")},label='Show Context')])
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



                '''nodes = cl.user_session.get("relevant_nodes")
                edges = cl.user_session.get("relevant_edges")

                fig = du.create_network_figure(nodes,target_edges=edges)
                element = cl.Plotly(name="network_graph", figure=fig, display="inline")'''


                chat_history = cl.user_session.get("chat_history")

                chat_history.append({
                                        "role": "assistant",
                                        "content": content,
                                    })

                cl.user_session.set("chat_history",chat_history)

