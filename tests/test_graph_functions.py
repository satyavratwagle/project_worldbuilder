from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path
import re
import ollama
import utils.prompts as prompts
import networkx as nx

#og_text = "Ryland Grace is a scientist who has been kicked out of academia because his peers don't believe him. He is later sent into space to save the world from Astrophage by Eva Stratt against his will. Later, he meets an alien named Rocky in outer space."

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='worldbuilding')



# Test Multihop

node_id = du.get_node_id('Shovzogr')
du.cluster_lore(node_id)
