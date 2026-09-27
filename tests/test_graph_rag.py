from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config)
du.load_embedding_model()

context = du.get_graph_rag_context('What is the relation between Marushar and Irnazogr?',strategy='find_path_between',categories=['location'])

print(context)