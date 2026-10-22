from utils.datastore_utils import DatastoreUtilities, get_code, get_name
#from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

#og_text = "Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky. Grace was once a brilliant biologist with wild theories that he hoped to prove true. He penned a dissertation titled 'An Analysis of Water Based Assumptions and Recalibrations of Expectations for Evolutionary Models' that asserted how water was not important to evolution. His paper labelled him an outcast in his field, but he was fired when he called the leading scholar in his field a 'staggering waste of carbon' at a UNESCO conference in Denmark. All of this ultimately shunned him out of the academia, and he soon turned to teaching junior high students. Grace is a tall, Caucasian male with an athletic build, sandy blonde hair, and blue eyes. Dr. Ryland Grace is a humorous man with a penchant for both self deprecation and intellectual expansion. He is great with children, possibly because of his own immaturity. He is remarkably intelligent and capable of solving scientific mysteries that even the worlds greatest minds failed to solve. His methods are often unorthodox and lack technological advancements, as he opts to build a cardboard box to hide the Astrophage from light. Grace is fascinated by the unknown and the universe. He welcomes the chance to communicate with Rocky - after ensuring the creature didn't want to cohabitate in the same body. Prior to his mission, Grace lacked bravery. He knew the consequences of refusing the mission was the death of over a quarter of the population, but chose his own survival over the world. Stratt, however, knew he was their only option. She had him captured, drugged, and put into a coma against his will. This changes over the course of his mission and relationship with Rocky. Rocky is an Eridian engineer from the planet Erid in the 40 Eridani solar system. He was sent to Tau Ceti with twenty-two other Eridian scientists/crewmates to save their star and planet from Astrophage. As an Eridian, he has no facial features and no sense of sight, instead using hearing and echolocation as his primary sense. He communicates using musical tones and chords. Rocky's true name in the Eridian language is a series of musical chords, but Grace calls him 'Rocky' for his rock-like appearance."
#"Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky."

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='multidigraph_test',current_graph='lore')
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
add_node("Mahamun","Agent","God",aliases=['The Almighty'])

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



