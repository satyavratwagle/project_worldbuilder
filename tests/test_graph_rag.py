from utils.datastore_utils import DatastoreUtilities
from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

#du = DatastoreUtilities(config,project_name='test',current_graph='lore')
#du.load_embedding_model()
sem = SemanticTools(config)


def get_clauses(text):

    clauses = sem.advanced_split_and_repair(text)
    print(f"\advanced_split_and_repair : {text} ---> {clauses}")
    return clauses

def explore_neighborhood(text):

    clauses = get_clauses(text)
    context, lore_nodes = du.get_graph_rag_context(clauses,strategy='explore_neighborhood',n_hops=1)
    print(f"\nexplore_neighborhood : {lore_nodes}")
    print(f"\nexplore_neighborhood : {context}")

def explore_path(text):

    clauses = get_clauses(text)
    context, lore_nodes = du.get_graph_rag_context(clauses,strategy='find_path_between')
    print(f"\nexplore_path : {lore_nodes}")
    print(f"\nexplore_path : {context}")

def propagate(node,hops=2,direction='forward'):
    nodes_ahead = du.propagate_plot_nodes(node,hops,'forward')
    nodes_behind = du.propagate_plot_nodes(node,hops,'backward')
    print(f"\nnodes_ahead : {nodes_ahead}")
    print(f"nodes_behind : {nodes_behind}")


texts = [
    #"Ryland Grace and Eva Stratt worked as scientists and they were sent to space to save the world.",
    #"Rocky was sent to Tau Ceti where he discovered the secrets of Astrophage ",
    #"Rocky was sent to Tau Ceti and discovered the secrets of Astrophage there.",
    #"Astrophage is an alien microbe and is responsible for the destruction of several planets.",
    #"Ryland Grace was a biologist, but then he was kicked out of academia.",
    #"Why was Ryland Grace kicked out of academia?",
    #"Why was Ryland Grace kicked out of academia and sent to space?",
    "Ryland Grace and Eva Stratt are dissimilar people who work in space.",
    "Rocky was sent to Tau Ceti where he discovered the secrets of Astrophage ",
    "Alice bought apples and oranges",
    "Mike and John are friends who meet often.",
    "Mike is a fireman, but then he was laid off.",
    "Mike is a fireman and is responsible for saving several people.",
    "Why was Mike laid off and sent to jail?",
    "How are Mike being a fireman and him being laid off related?",
    "What connects Mike being a fireman and him being laid off?",
    "How would you describe Mike and Josh?"

]

#explore_neighborhood("What are the similarities between Grace and Rocky?")

for text in texts:
    clauses = get_clauses(text)

'''
for node,node_data in du.knowledge_graph.nodes(data=True):
    if(node_data['layer']=='lore'):
        print(node,[neighbor for neighbor in du.knowledge_graph.predecessors(node)])
'''