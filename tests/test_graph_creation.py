from utils.datastore_utils import DatastoreUtilities, get_code, get_name
#from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter


with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='multidigraph_test',current_graph='lore')
#du.load_embedding_model()
#sem = SemanticTools(config)
#sem.load_extraction_model()

sentences = ["Grace is an academic.","Grace and Rocky are friends","Grace and Rocky are companions","Rocky and Grace are friends","Grace and Stratt are not friends.","Grace is a scientist."]

def check_graph():

    print('\nGraph details : ',du.knowledge_graph)


def add_node(node_name,node_type,node_subtype,**metadata):

    node_id = du.add_node(node_name,node_type,node_subtype,**metadata)
    if(node_id):
        print(du.knowledge_graph.nodes[node_id],'\n')
    else:
        print("No node added!\n")

def remove_node(node_name):
    node_data = du.remove_node(node_name)
    print(f"\nRemoved Node : {node_data}\n")

def remove_edge(head,tail,type=None,sub_type=None):
    edge_data = du.remove_edge(head,tail,type,sub_type)
    print(f"\nRemoved Edge : {edge_data}")

def add_edge(head, tail, edge_type, edge_subtype,edge_text):
    edge = du.add_edge(head,tail,edge_type,edge_subtype,edge_text)
    print(f"\nAdded Edge : {edge}")

def get_edges(node):
    edge_data = du.get_all_edges(node)

    for key,data in edge_data:
        print(f"{data['type']} ({", ".join(data['tags'])}) : {data['text']}")

def lookup_edge_by_type(type,tag=None):
    edges = du.get_edges_of_type(type,tag)
    print(f"\nFound Edge : {edges}")
    for edge in edges:
        print(f"{edge} : {du.knowledge_graph.edges[edge]['text']}")

def lookup_node_by_type(type,tag=None):
    nodes = du.get_nodes_of_type(type,tag)
    print(f"\nFound Node : {nodes}")
    for node in nodes:
        print(f"{node} : {du.knowledge_graph.nodes[node]['text']}")

def save_graph():
    du.save_graph()

def reload_lookups():
    du.reload_lookups()

def add_metadata(node_name,**metadata):
    info = du.add_metadata(node_name,**metadata)

# Graph Query functions

# Dataset functions
def update_dataset():
    dataset,index = du.update_faiss_dataset()
    du.overwrite_faiss_dataset(dataset,index)

def check_dataset():

    print(du.dataset)
    for row in du.dataset:
        print(f"{row['node_id']} = {row['text']}")


# Add basic node
#add_node("Mahamun","Agent","God",aliases=['The Almighty'])

# Add duplicate nodes - Expect second one to not be added.
#add_node("Mahamun","Agent","God")
#add_node("Mahamun","Agent","God")

# Add node with metadata
#add_node("Alvar Iral-zid","Agent","Character",aliases=['Alvie'])

# Add metadata separately

# Remove node
#add_node("Mahamun","Agent","God")
#add_node("Mahamun","Agent","God")
#remove_node("Mahamun")
#add_node("Mahamun","Agent","God")

# Add edge
#add_node("Mahamun","Agent","God")
#add_node("Marushar","Location","Desert")
#add_edge("Mahamun","Marushar","Spatial","Birthplace","Marushar is the birthplace of Mahamun")
#add_edge("Mahamun","Marushar","Spatial","Homeland","Mahamun hails from Marushar")

# Remove Edge
#add_node("Mahamun","Agent","God")
#add_node("Marushar","Location","Desert")
#add_edge("Mahamun","Marushar","Spatial","Birthplace","Marushar is the birthplace of Mahamun")
#add_edge("Mahamun","Marushar","Spatial","Homeland","Mahamun hails from Marushar")
#add_edge("Mahamun","Marushar","Structure","Temple","Mahamun built a temple in Marushar")
#remove_edge("Mahamun","Marushar","Spatial")
#check_graph()

# Retrieve Edge by Type
add_node("Mahamun","Agent","God")
add_node("Marushar","Location","Desert")
add_node("Shovzogr","Location","Ocean")
add_edge("Mahamun","Marushar","Spatial","Birthplace","Marushar is the birthplace of Mahamun")
add_edge("Mahamun","Marushar","Spatial","Homeland","Mahamun hails from Marushar")
add_edge("Mahamun","Marushar","Structure","Temple","Mahamun built a temple in Marushar")
add_edge("Shovzogr","Mahamun","Function","Crosses","Shovzogr was crossed by Mahamun.")
get_edges("Mahamun")

#lookup_edge_by_type("Spatial","Homeland")
#lookup_node_by_type("Location","Ocean")
#du.check_edges_for_redundancy("Mahamun","Marushar","Mahamun danced in the moonlight in Marushar")

