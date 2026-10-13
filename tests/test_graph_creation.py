from utils.datastore_utils import DatastoreUtilities
#from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

og_text = "Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky. Grace was once a brilliant biologist with wild theories that he hoped to prove true. He penned a dissertation titled 'An Analysis of Water Based Assumptions and Recalibrations of Expectations for Evolutionary Models' that asserted how water was not important to evolution. His paper labelled him an outcast in his field, but he was fired when he called the leading scholar in his field a 'staggering waste of carbon' at a UNESCO conference in Denmark. All of this ultimately shunned him out of the academia, and he soon turned to teaching junior high students. Grace is a tall, Caucasian male with an athletic build, sandy blonde hair, and blue eyes. Dr. Ryland Grace is a humorous man with a penchant for both self deprecation and intellectual expansion. He is great with children, possibly because of his own immaturity. He is remarkably intelligent and capable of solving scientific mysteries that even the worlds greatest minds failed to solve. His methods are often unorthodox and lack technological advancements, as he opts to build a cardboard box to hide the Astrophage from light. Grace is fascinated by the unknown and the universe. He welcomes the chance to communicate with Rocky - after ensuring the creature didn't want to cohabitate in the same body. Prior to his mission, Grace lacked bravery. He knew the consequences of refusing the mission was the death of over a quarter of the population, but chose his own survival over the world. Stratt, however, knew he was their only option. She had him captured, drugged, and put into a coma against his will. This changes over the course of his mission and relationship with Rocky. Rocky is an Eridian engineer from the planet Erid in the 40 Eridani solar system. He was sent to Tau Ceti with twenty-two other Eridian scientists/crewmates to save their star and planet from Astrophage. As an Eridian, he has no facial features and no sense of sight, instead using hearing and echolocation as his primary sense. He communicates using musical tones and chords. Rocky's true name in the Eridian language is a series of musical chords, but Grace calls him 'Rocky' for his rock-like appearance."
#"Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky."

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='test',current_graph='lore')
#sem = SemanticTools(config)
#sem.load_extraction_model()

sentences = ["Grace is an academic.","Grace and Rocky are friends","Grace and Rocky are companions","Rocky and Grace are friends","Grace and Stratt are not friends.","Grace is a scientist."]

def check_graph():

    print('\nGraph details : ',du.knowledge_graph)
    print('Lore Node Index  : ',du.knowledge_graph.graph['lore_idx'])
    print('Plot Node Index  : ',du.knowledge_graph.graph['plot_idx'])
    print('Topic Node Index : ',du.knowledge_graph.graph['topic_idx'])

    print()
    for node_id in du.knowledge_graph.nodes():
        print(f"{node_id} : {du.knowledge_graph.nodes[node_id]['desc']}")
    print()

    for head,tail,text in du.knowledge_graph.edges(data=True):
        print(f"{du.knowledge_graph.nodes[head]['desc']} --> {du.knowledge_graph.nodes[tail]['desc']} : {text}")
    print()

    print(du.node_lookup)


def add_node(node_name,node_type,node_label='misc',**kwargs):

    node_id = du.add_node(node_name,node_type,node_label,**kwargs)
    print(f"\nAdded Node : {node_id} - {du.knowledge_graph.nodes[node_id]}")

def remove_node(node_name):
    node_data = du.remove_node(node_name)
    print(f"\nRemoved Node : {node_data}")

def remove_edge(head,tail):
    edge_data = du.remove_edge(head,tail)
    print(f"\nRemoved Edge : {edge_data}")

def add_edge(head, tail, edge_label):
    edge = du.add_edge(head,tail,edge_label)
    print(f"\nAdded Edge : {edge}")

def save_graph():
    du.save_graph()

def reload_lookups():
    du.reload_lookups()

def lookup_metadata(node_name,key):
    info = du.get_node_metadata(node_name,key)
    print(f"\nMetadata for {key} : {info}")

# Graph Query functions

def extrapolate():

    output = du.extrapolate_descriptions()
    print(f"\nextrapolate_descriptions : {output}")

    output = du.extrapolate_involvements()
    print(f"\nextrapolate_involvements : {output}")

    output = du.extrapolate_relationships()
    print(f"\nextrapolate_relationships : {output}")

def filter_graph(layers):
    subgraph = du.filter_graph(layers)
    print(f"\nfilter_graph : {subgraph}")

def find_path(source,target,layer):
    path = du.find_path_between(source,target,layer)
    print(f"\nfind_path : {path}")

def explore(node,hops):
    neighborhood = du.explore_neighborhood(node,hops)
    lore = du.get_lore_from_nodes(neighborhood)

def query_dataset(query):
    results = du.find_similar_nodes(query,k=1)
    for node_id, score, text in results:
        print(f"{score} : {text} ({node_id})")

def order_plot(before,after):
    edge = du.sequence_scenes(before,after)
    print(f"\norder_plot : {edge}")

# Dataset functions
def update_dataset():
    dataset,index = du.update_faiss_dataset()
    du.overwrite_faiss_dataset(dataset,index)

def check_dataset():

    print(du.dataset)
    for row in du.dataset:
        print(f"{row['node_id']} = {row['text']}")

phase_construct_graph = True
phase_overwrite_dataset = True

phase_test_graph = True
phase_test_dataset = False

check_graph()
check_dataset()

if(phase_construct_graph):

    # Save the topics first
    
    add_node("Ryland Grace","topic","Character",summary='Summary for Ryland Grace',aliases=['Grace'])
    add_node("Eva Stratt","topic","Character",summary='Summary for Eva Stratt',aliases=['Stratt'])
    add_node("Rocky","topic","Character",summary='Summary for Rocky',aliases=['Rocky'])

    # Always reload lookups after saving topics
    reload_lookups()

    add_node("Ryland Grace was a biologist.","lore")
    add_node("Stratt and Grace are not similar in personality.","lore")
    add_node("Rocky and Grace are similar in personality.","lore")

    add_node("Ryland Grace was shunned from academia","plot")
    add_node("Grace became a high school teacher.","plot")
    add_node("Eva Stratt recruited Grace.","plot")
    add_node("Grace was sent to space on the Hail Mary mission.","plot")
    add_node("Grace met Rocky in space.","plot")

    #remove_node('Ryland Grace')
    #remove_node("Ryland Grace was sent to space on the Hail Mary mission.")

    reload_lookups()
    save_graph()

if(phase_test_graph):
    #extrapolate()
    #save_graph()
    #filter_graph(['topic'])

    #du.load_embedding_model()
    #find_path("Grace was kicked out of the academic circles.","In space, Grace found Rocky.","plot")

    add_node("Erid","topic","Location",summary='Summary for Erid',aliases=['Erid'])
    add_node("Eridians","topic","Character",summary='Summary for Eridians',aliases=['Eridians'])
    add_node("Adrian","topic","Character",summary='Summary for Adrian',aliases=['Adrian'])
    reload_lookups()

    add_node("Rocky is one of the Eridians.","lore")
    add_node("Eridians live on Erid.","lore")
    add_node("Erid is also called Adrian.","lore")

    extrapolate()

    explore('Adrian',hops=5)

if(phase_overwrite_dataset):
    du.load_embedding_model()
    update_dataset()
    check_dataset()

if(phase_test_dataset):
    #du.load_embedding_model()
    order_plot("Grace was kicked out of the academic circles.","Grace started to teach in school.")
    order_plot("Grace started to teach in school.","Stratt approached Grace with a job.")
    order_plot("Stratt approached Grace with a job.","Grace was launched on a shuttle.")
    order_plot("Grace was launched on a shuttle.","In space, Grace found Rocky.")
    save_graph()
    update_dataset()
    check_dataset()


    