from utils.datastore_utils import DatastoreUtilities
from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

og_text = "Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky. Grace was once a brilliant biologist with wild theories that he hoped to prove true. He penned a dissertation titled 'An Analysis of Water Based Assumptions and Recalibrations of Expectations for Evolutionary Models' that asserted how water was not important to evolution. His paper labelled him an outcast in his field, but he was fired when he called the leading scholar in his field a 'staggering waste of carbon' at a UNESCO conference in Denmark. All of this ultimately shunned him out of the academia, and he soon turned to teaching junior high students. Grace is a tall, Caucasian male with an athletic build, sandy blonde hair, and blue eyes. Dr. Ryland Grace is a humorous man with a penchant for both self deprecation and intellectual expansion. He is great with children, possibly because of his own immaturity. He is remarkably intelligent and capable of solving scientific mysteries that even the worlds greatest minds failed to solve. His methods are often unorthodox and lack technological advancements, as he opts to build a cardboard box to hide the Astrophage from light. Grace is fascinated by the unknown and the universe. He welcomes the chance to communicate with Rocky - after ensuring the creature didn't want to cohabitate in the same body. Prior to his mission, Grace lacked bravery. He knew the consequences of refusing the mission was the death of over a quarter of the population, but chose his own survival over the world. Stratt, however, knew he was their only option. She had him captured, drugged, and put into a coma against his will. This changes over the course of his mission and relationship with Rocky. Rocky is an Eridian engineer from the planet Erid in the 40 Eridani solar system. He was sent to Tau Ceti with twenty-two other Eridian scientists/crewmates to save their star and planet from Astrophage. As an Eridian, he has no facial features and no sense of sight, instead using hearing and echolocation as his primary sense. He communicates using musical tones and chords. Rocky's true name in the Eridian language is a series of musical chords, but Grace calls him 'Rocky' for his rock-like appearance."
#"Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky."


extract_coreferences = False
add_edge = False
update_summaries = False
get_rag_context = False

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='demo',current_graph='lore',is_core_graph=True)
du.load_embedding_model()
sem = SemanticTools(config)
#sem.load_extraction_model()

sentences = ["Grace is an academic.","Grace and Rocky are friends","Grace and Rocky are companions","Rocky and Grace are friends","Grace and Stratt are not friends.","Grace is a scientist."]

idxs_to_keep = du.resolve_text(sentences)
print([sentences[idx] for idx in idxs_to_keep])

if(extract_coreferences):
    selections = ['Rocky','Grace','Stratt']
    doc,entity_coreferences = sem.get_coref_clusters(og_text,selections)
    aliases = sem.find_aliases(doc,entity_coreferences)
    resolved_text = sem.resolve_coreferences(doc,entity_coreferences)

    for topic,aliases in aliases.items():
        du.add_node(topic,"character",aliases=aliases)

    print(du.knowledge_graph)

if(add_edge):
    du.add_edge('grace','rocky',"Grace and Rocky are friends.")
    du.add_edge('grace','stratt',"Grace and Stratt are acquaintances.")
    du.save_graph()

if(update_summaries):
    for node in du.knowledge_graph.nodes(data=True):
        print(node)
        du.add_to_node_summary(node[0],'location',f"This is a new location for {node[1]['name']}")
        du.add_to_node_summary(node[0],'event',f"This is a new event for {node[1]['name']}")
        du.add_to_node_summary(node[0],'process',f"This is a new process for {node[1]['name']}")
    du.save_graph()
    d,i, = du.filter_dataset()
    du.overwrite_faiss_dataset(d,i)

if(get_rag_context):
    context, nodes, edges = du.get_graph_rag_context("Eva Stratt and Rocky might be friends.",strategy='find_path_between')
    
