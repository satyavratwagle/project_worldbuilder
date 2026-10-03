from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path
import re
import ollama
import utils.prompts as prompts
import networkx as nx

og_text = "Ryland Grace is a scientist who has been kicked out of academia because his peers don't believe him. He is later sent into space to save the world from Astrophage by Eva Stratt against his will. Later, he meets an alien named Rocky in outer space."

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='demo',current_graph='lore',is_core_graph=True)

# Test Adamic Adar index

for node1 in du.knowledge_graph.nodes():
	for node2 in du.knowledge_graph.nodes():

		preds = nx.adamic_adar_index(nx.Graph(du.knowledge_graph), [(node1,node2)])

		for u,v,p in preds:
			if(not(u==v) and not nx.Graph(du.knowledge_graph).has_edge(u,v)):
				print(f"{du.get_node_name(u)} & {du.get_node_name(v)} = {p}")