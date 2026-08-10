import networkx as nx
import ahocorasick
import os
import shutil
import json
import networkx as nx
import matplotlib.pyplot as plt
import re
from pathlib import Path

def uppercase(text):
    return text[0].upper()+text[1:]

class JSONUtils():

    def __init__(self):

        self.jsonstore_dir = '/Users/satyawagle/Projects/LangChain/llm_chatbot/data/json_store'
        self.faiss_dataset_path = './data/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = "./data/FAISS_store/worldbuilding_dataset.faiss"

    def overwrite_faiss_dataset(self,new_dataset,cosine_index):

        dataset_path = self.faiss_dataset_path
        faiss_path = self.faiss_index_path

        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)

        new_dataset.save_to_disk(dataset_path)
        new_dataset.add_faiss_index(
            column="embeddings", 
            custom_index=cosine_index
        )
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        new_dataset.save_faiss_index("embeddings", faiss_path)
        # 5. Persist the index and documents locally
        #faiss_vectorstore.save_local("data/FAISS_store/faiss_json_index")
        print("FAISS datastore successfully created and saved!")

    def load_files(self):

        # Ensure tags field exists in raw json files
        files = os.listdir(self.jsonstore_dir)
        
        documents = []
        for filename in files:

            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 2. Parse JSON data into LangChain Document objects
                # Adjust keys ('text', 'metadata') based on your JSON structure    

                keys = data['data'].keys()

                text = []
                for key in keys:
                    if(len(data['data'][key])>0):
                        text = f"{uppercase(key)} of {uppercase(data['name'])} : {' '.join(data['data'][key])}"
                        metadata = {'name':data['name'],'type':data['type'],'tags':data['tags']+[key],'related':data['related']}
                        #documents.append(Document(page_content=text, metadata=metadata))
                        documents.append({'text':text,
                            'name':metadata['name'],
                            'id': f"{metadata['name']}.{key}",
                            'type':metadata['type'],
                            'tags':metadata['tags'],
                            'related':metadata['related']})

        return documents

    def create_knowledge_graph(self):

        graph = nx.DiGraph()
        files = os.listdir(self.jsonstore_dir)

        # Pass 1: Map every filename to its entity 'name'

        documents_lookup = {}
        filename_to_name = {}
        for filename in files:
            if(filename.endswith('.json')):
                file_path = os.path.join(self.jsonstore_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    name = f"{data.get('name')}.overview"
                    if name:
                        filename_to_name[filename] = name
                        documents_lookup.setdefault(name,f"Overview of {uppercase(data.get('name'))} : {' '.join(data['data']['overview'])}")


        # Pass 2: Build graph edges
        for filename in files:
            if(filename.endswith('.json')):
                file_path = os.path.join(self.jsonstore_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                name = data.get('name', 'Unknown')
                doc_type = data.get('type', 'Unknown')
                tags = data.get('tags', [])

                # Add node to graph
                for key in data['data'].keys():
                    graph.add_node(f"{name}.{key}", type=doc_type)

                    related_files = data['related'][key]
                    # Build edges using the explicit 'related' field schema
                    for rel_file in related_files:
                        target_name = filename_to_name.get(rel_file)
                        graph.add_edge(f"{name}.{key}", target_name)

        return graph, documents_lookup, filename_to_name

    def update_links(self):

        # Updates the links within JSON files.

        files = os.listdir(self.jsonstore_dir)
        topics = []
        for filename in files:
            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                topics.append(data['name'])

        A = ahocorasick.Automaton()
        for name in topics:
          A.add_word(name.lower(), name)  # store original name as value
        A.make_automaton()

        for filename in files:
            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                keys = data['data'].keys()
                source_name = data['name']
                data['related'] = dict()

                for key in keys:

                    found_targets = set()
                    text = ' '.join(data['data'][key])

                    data['related'][key] = []

                    # Scan the text in a single efficient pass
                    for end_index, original_name in A.iter(text.lower()):
                        if original_name != source_name.lower():
                            found_targets.add(original_name)

                    for target in found_targets:
                        targetname = target.title().lower()
                        targetname = re.sub(r'[^a-zA-Z0-9]', '_', targetname)
                        data['related'][key].append(targetname+'.json')
                        

                with open(self.jsonstore_dir+'/'+filename, "w") as f:
                    data = json.dump(data, f, indent=4)

    def get_json_data(self):

        # Updates the tags within JSON files.

        files = os.listdir(self.jsonstore_dir)
        json_text = dict()
        for filename in files:
            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                json_text[data['name']] = []

                keys = data['data'].keys()

                for key in keys:
                    json_text[data['name']].append(' '.join(data['data'][key]))

                json_text[data['name']] = ''.join(json_text[data['name']])

        return json_text

    def save_json(self,name,data):
        entity = name.lower()
        filename = re.sub(r'[^a-zA-Z0-9]', '_', entity)
        with open(self.jsonstore_dir+'/'+filename+'.json', "w") as file:
            json.dump(data, file, indent=4)

    def load_json(self,name):

        entity = name.lower()
        filename = re.sub(r'[^a-zA-Z0-9]', '_', entity)
        exists = Path(f"{self.jsonstore_dir}/{filename}.json").exists()
        if(exists):
            with open(f"{self.jsonstore_dir}/{filename}.json", "r") as file:
                data = json.load(file)
        else:
            data = None

        return exists, data

    def get_alias_map(self):

        alias_map = dict()
        files = os.listdir(self.jsonstore_dir)

        for filename in files:
            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                for alias in data['aliases']:
                    alias_map[alias.lower()] = data['name'].lower()

        return alias_map

    def get_blurb_map(self):

        blurb_map = dict()
        files = os.listdir(self.jsonstore_dir)

        for filename in files:
            if(filename.split('.')[-1]=='json'):

                with open(self.jsonstore_dir+'/'+filename, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if(len(data['blurb'].strip())>0):
                    blurb_map[data['name']] = data['blurb']

        return blurb_map

