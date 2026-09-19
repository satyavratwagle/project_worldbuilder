import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
from utils.semantic import SemanticTools
import json
import copy
import torch
import numpy as np

@cl.on_chat_start
async def on_chat_start():

    with open('config.json', "r", encoding="utf-8") as f:
        config = json.load(f)

    with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
        wiki_schema = json.load(f)
    
    du = DatastoreUtilities(config)
    
    sem = SemanticTools(config)
    sem.load_zsc_model("knowledgator/gliclass-modern-base-v3.0")

    def process_wiki_schema(src_dict,topic=None):
        wiki_dict = copy.deepcopy(src_dict)
        if(topic):
            for key,value in wiki_dict.items():
                wiki_dict[key] = wiki_dict[key].replace(f'[TOPIC]',topic)
        return wiki_dict

    for node in du.get_all_nodes():

        topic       = node['id']
        topic_type  = du.knowledge_graph.nodes[node['id']]['type']
        topic_summary = du.knowledge_graph.nodes[node['id']]['summary']

        schema = process_wiki_schema(wiki_schema[topic_type.lower()],topic)
        labels = [f"{schema[key]}" for key in schema.keys()]
        schema_keys = list(schema.keys())

        node_wiki = dict()
        for key in schema.keys():
            node_wiki[key] = []
        
        for test_edge in du.knowledge_graph.edges(node['id'],keys=True,data=True):

            edge_text = test_edge[3]['desc']
            results = sem.zero_shot_classification(edge_text,labels,context=[topic_summary],threshold=0.0)

            # Get best label
            best_idx = [r[1] for r in results[0]].index(max([r[1] for r in results[0]]))
            #print(best_idx)
            node_wiki[schema_keys[best_idx]].append(edge_text)

            #print(results)
            #print(f"{edge_text} :",[f"{r[0]} : {r[1]:.3f}" for r in results[0]])
            #print()

        print(node)
        print(node_wiki)
