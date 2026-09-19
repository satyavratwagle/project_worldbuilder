from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
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
import copy

# USE THIS LATER
def process_prompt(src_dict,topic=None):

    wiki_dict = copy.deepcopy(src_dict)
    
    if(topic):
        for key,value in wiki_dict.items():
            wiki_dict[key] = wiki_dict[key].replace(f'[TOPIC]',topic)

    return wiki_dict


with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
    wiki_schema = json.load(f)

model = GLiClassModel.from_pretrained("knowledgator/gliclass-modern-base-v3.0")
tokenizer = AutoTokenizer.from_pretrained("knowledgator/gliclass-modern-base-v3.0")

pipeline = ZeroShotClassificationPipeline(
    model, tokenizer, classification_type='multi-label', device='mps'
)

with open('config.json', "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    config = json.load(f)

du = DatastoreUtilities(config)

STORE_DIR = f"{config['data_dir']}/FAISS_store/"

topic = "Marushar"
type = 'location'
node_info = du.get_node_info(topic)
#print(node_info['edge_info'])

if(False):
    dataset = datasets.load_from_disk(os.path.join(STORE_DIR, "worldbuilding_dataset"))
    dataset.load_faiss_index("embeddings", os.path.join(STORE_DIR, "worldbuilding_dataset.faiss"))
    embed_model = SentenceTransformer("BAAI/bge-m3")
    du.load_embedding_model(embed_model,dataset)

wiki_schema[type] = process_prompt(wiki_schema[type],topic)

prompt = f"Classify the text related to {topic} using these specific definitions:\n{"\n".join(key+" : "+wiki_schema[type][key] for key in wiki_schema[type].keys())}"
print(prompt)

# Extract edge and the context of the nodes attached.
all_text = []
for topic,neighbor,key,text in node_info['edge_info']:

    neighbor = neighbor if not(neighbor==topic) else None

    context = "--- CONTEXT BEGINS ---\n"
    if(neighbor):
        n_context = du.get_node_summary(neighbor)
        if(n_context):
            context += n_context
    context += "\n--- CONTEXT ENDS ---\n\n"

    all_text.append(context+text)

'''context = "CONTEXT: Hatyaars are a group of mercenary killers.\n\n"
all_text = [context+text for text in [e[3] for e in node_info['edge_info']]]'''
labels = list(wiki_schema[type].keys())
results = pipeline(all_text, labels,threshold=0.1)

for idx in range(len(all_text)):
    print(all_text[idx].split('\n\n')[-1])
    print([f"{d['label']} : {d['score']:.3f}" for d in results[idx]])
    print()