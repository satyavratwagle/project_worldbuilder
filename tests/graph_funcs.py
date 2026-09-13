import json
import torch
import numpy as np

import re
import sys
import os
import networkx as nx
import matplotlib.pyplot as plt
import ahocorasick
import string
import ast
import datasets
from sentence_transformers import SentenceTransformer

from pathlib import Path
from utils.datastore_utils import DatastoreUtilities
import time

with open('config.json', "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    config = json.load(f)

du = DatastoreUtilities(config)

STORE_DIR = f"{config['data_dir']}/FAISS_store/"


topics = ['Mahamun','Archipelago','Gng']
print('\n'.join([summary for summary in du.get_node_summaries(topics) if summary]))

if(False):
    dataset = datasets.load_from_disk(os.path.join(STORE_DIR, "worldbuilding_dataset"))
    dataset.load_faiss_index("embeddings", os.path.join(STORE_DIR, "worldbuilding_dataset.faiss"))

    #print([doc['topic'] for doc in dataset])
    embed_model = SentenceTransformer("BAAI/bge-m3")
    du.load_embedding_model(embed_model,dataset)

#new_edges = du.find_new_edges()
#print(new_edges)

#dataset,index = du.update_faiss_dataset()
#du.overwrite_faiss_dataset(dataset,index)

#context,edges = du.get_graph_rag_context("When volcanos explode, Ambar clouds are blown into the sky. They hurt the Locana.")
#for e in edges:
#    print(e)