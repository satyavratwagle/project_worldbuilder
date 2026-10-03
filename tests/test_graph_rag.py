from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='demo')
#du.load_embedding_model()

#context = du.get_graph_rag_context('What is the relation between Marushar and Irnazogr?',strategy='find_path_between',categories=['location'])

matched_text, resolved_text = du.check_for_similar_nodes("Tau Cety is a bad thing.")

print(resolved_text)
for source,target in matched_text:
    print(f"{source} ---> {target}")