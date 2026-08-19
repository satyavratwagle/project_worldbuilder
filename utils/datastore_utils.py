import torch
from gliner import GLiNER
import utils.preprocessing as pre
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from sentence_transformers import SentenceTransformer
import datasets
import networkx as nx
from datasets import Dataset,concatenate_datasets
import faiss
import os
from rank_bm25 import BM25Okapi
import json
from fastcoref import FCoref
import spacy
import psutil
import gc
import shutil
import re
import ahocorasick
from pathlib import Path
from datetime import datetime

# Patch to prevent AttributeError with older packages on Transformers 5.x
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr
def uppercase(text):
    return text[0].upper()+text[1:]

class DatastoreUtilities():

    def __init__(self,config):

        self.store_path = config['data_dir']
        self.jsonstore_dir = f'{self.store_path}/json_store'
        self.faiss_dataset_path = f'{self.store_path}/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = f"{self.store_path}/FAISS_store/worldbuilding_dataset.faiss"

    def update_local_dataset(self):

        documents = self.load_files()
        new_dataset, index = self.create_new_dataset(documents)
        
        # Filter out existing data from old dataset
        old_dataset = datasets.load_from_disk(os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset"))

        filtered_dataset = old_dataset
        filtered = False
        for doc in new_dataset:
            filtered_dataset = filtered_dataset.filter(lambda example: example["name"] != doc['name'])
            filtered = True

        # Remove any deleted files
        filenames = list(set([doc['name'] for doc in old_dataset]))
        excluded_names = []
        for name in filenames:
            filename = re.sub(r'[^a-zA-Z0-9]', '_', name)
            if(not(os.path.exists(f'{self.jsonstore_dir}/{filename}.json'))):
                excluded_names.append(name)

        for name in excluded_names:
            filtered_dataset = filtered_dataset.filter(lambda example: example["name"] != name)
            filtered = True

        if(filtered):

            # Update links here.
            self.update_links()

            # Append datasets.
            updated_dataset = concatenate_datasets([filtered_dataset,new_dataset])

            self.overwrite_faiss_dataset(updated_dataset,index)
            self.reload_faiss_dataset()

    def load_embedding_model(self,embed_model,dataset):
        self.text_embedding_model = embed_model
        self.dataset = dataset

    def create_new_dataset(self,documents):

        new_dataset = Dataset.from_list(documents)

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}
            
        embedding_dim = self.text_embedding_model.get_embedding_dimension()
        cosine_index = faiss.IndexFlatIP(embedding_dim)
        new_dataset = new_dataset.map(embed_text, batched=True, batch_size=8)

        return new_dataset,cosine_index

    def overwrite_faiss_dataset(self,new_dataset,cosine_index):

        temp_dataset_path = self.faiss_dataset_path+'_temp'
        temp_faiss_path = temp_dataset_path+'.faiss'

        # Save to temporary location
        new_dataset.save_to_disk(temp_dataset_path)
        new_dataset.add_faiss_index(
            column="embeddings", 
            custom_index=cosine_index
        )
        if os.path.exists(temp_faiss_path):
            os.remove(temp_faiss_path)
        new_dataset.save_faiss_index("embeddings", temp_faiss_path)

        # Delete old dataset if exists
        if os.path.exists(self.faiss_dataset_path):
            shutil.rmtree(self.faiss_dataset_path)

        # Rename temporary location
        os.rename(temp_dataset_path,self.faiss_dataset_path)
        os.rename(temp_faiss_path,self.faiss_index_path)

        print("FAISS datastore successfully created and saved!")

    def reload_faiss_dataset(self):

        self.dataset = datasets.load_from_disk(os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset"))
        self.dataset.load_faiss_index("embeddings", os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset.faiss"))

    def get_most_relevant_file(self,query,threshold,k):

        query_vector = self.embed_text(query)
        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=k)
    
        if scores[0] >= threshold:
            return examples
        else:
            return []

    def get_graph_rag_context(self,query,graph,documents_lookup,threshold=0.4,k=10,hops=1):

        # need self.knowledge_graph, self.documents_lookup

        query_vector = self.embed_text(query)
        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=k)

        seed_results = []
        for i in range(len(scores)):
            if(scores[i] >= threshold and len(examples['text'][i].split(':')[-1].strip())>0):
                seed_results.append({'name':examples['name'][i],'text':examples['text'][i],'id':examples['id'][i]})

        retrieved_contexts = dict()
        retrieved_contexts['direct'] = []
        retrieved_contexts['indirect'] = []

        retrieved_context_ids = dict()
        retrieved_context_ids['direct'] = []
        retrieved_context_ids['indirect'] = []

        seen_names = set()

        for doc in seed_results:
            seen_names.add(doc["id"].lower())


        for doc in seed_results:
            name = doc["id"].lower()
            
            # Add primary seed document if not already added
            retrieved_contexts['direct'].append(doc["text"])
            retrieved_context_ids['direct'].append(doc["id"])
            
            # Step 2: Graph Expansion (Traverse 1 or more hops via the 'related' field)
            if graph.has_node(name):
                # Find all connected nodes within N hops
                neighbors = nx.single_source_shortest_path_length(graph, name, cutoff=hops)
                for neighbor_name in neighbors:
                    if neighbor_name != name and neighbor_name not in seen_names:
                        seen_names.add(neighbor_name)
                        # Pull all text chunks belonging to the related file/entity
                        neighbor_info = documents_lookup.get(neighbor_name, [])
                        if(len(neighbor_info.split(':')[-1].strip())>0):
                            retrieved_contexts['indirect'].append(neighbor_info)
                            retrieved_context_ids['indirect'].append(neighbor_name)

        tokenized_context_docs = [text.lower().split() for text in retrieved_contexts['direct']+retrieved_contexts['indirect']]
        if(len(tokenized_context_docs)>0):
            self.bm25_model = BM25Okapi(tokenized_context_docs)
            tokenized_query = query.lower().split()
            doc_scores = self.bm25_model.get_scores(tokenized_query)

        return retrieved_contexts,retrieved_context_ids

    def embed_text(self,text):
        return self.text_embedding_model.encode(text, normalize_embeddings=True)

    def get_cosine_similarity(self,emb1,emb2):
        dot_product = np.dot(emb1, emb2)
        norm_vec1 = np.linalg.norm(emb1)
        norm_vec2 = np.linalg.norm(emb2)

        similarity = dot_product / (norm_vec1 * norm_vec2)

        return similarity

    def load_files(self):
        # Return only the files that have been updated

        dataset_update_time = os.path.getmtime(self.faiss_dataset_path)

        # Ensure tags field exists in raw json files
        files = os.listdir(self.jsonstore_dir)
        
        documents = []
        for filename in files:

            if(filename.split('.')[-1]=='json'):

                file_update_time = os.path.getmtime(f'{self.jsonstore_dir}/{filename}')

                if(file_update_time>dataset_update_time):

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

