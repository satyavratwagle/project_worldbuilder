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


for edge in du.knowledge_graph.edges(keys=True,data=True):
    print(edge)

#du.save_graph()
