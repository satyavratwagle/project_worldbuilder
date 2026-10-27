import torch
import numpy as np
import datasets
import networkx as nx
from datasets import Dataset,Features,Value, concatenate_datasets
import faiss
import os
import json
import re
from pathlib import Path
from datetime import datetime, date
import matplotlib.pyplot as pltk
import difflib
import time
import random
import shutil
import plotly.graph_objects as go
import textwrap
from rapidfuzz import process, fuzz
import itertools
import string
import umap
from sklearn.neighbors import NearestNeighbors
import ahocorasick
import math
import base64

# Patch to prevent AttributeError with older packages on Transformers 5.x
'''
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)

torch.nn.Module.__getattr__ = _patched_getattr
'''

def uppercase(text):
    return text[0].upper()+text[1:]

def slugify_key(text: str) -> str:

    # Convert to lowercase
    text = text.lower()
    # Replace spaces and hyphens with underscores
    text = re.sub(r'[\s\-]+', '_', text)
    # Remove all characters that aren't alphanumeric or underscores
    text = re.sub(r'[^a-z0-9_]', '', text)
    # Strip leading/trailing underscores
    return text.strip('_')

# Generating alphanumeric node ids
ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase

def get_code(text: str) -> str:
  byte_data = text.encode("utf-8")

  # Count leading zero bytes to preserve them during conversion
  leading_zeros = 0
  for byte in byte_data:
    if byte == 0:
      leading_zeros += 1
    else:
      break

  # Convert bytes to a massive integer
  num = int.from_bytes(byte_data, "big")
  if num == 0:
    return ALPHABET[0] * max(1, leading_zeros)

  # Convert integer to Base62
  res = []
  while num > 0:
    num, rem = divmod(num, 62)
    res.append(ALPHABET[rem])

  encoded = "".join(reversed(res))
  # Prefix leading zeros back into the string representation
  return (ALPHABET[0] * leading_zeros) + encoded

def get_name(encoded_str: str) -> str:
  # Count leading '0' characters
  leading_zeros = 0
  for char in encoded_str:
    if char == ALPHABET[0]:
      leading_zeros += 1
    else:
      break

  # Convert Base62 string back to an integer
  num = 0
  for char in encoded_str:
    num = num * 62 + ALPHABET.index(char)

  if num == 0:
    byte_data = b"\x00" * leading_zeros
  else:
    byte_len = (num.bit_length() + 7) // 8
    byte_data = (b"\x00" * leading_zeros) + num.to_bytes(byte_len, "big")

  return byte_data.decode("utf-8")

def generate_alphanumeric_code(length=4):
    # Combine ascii letters (a-z, A-Z) and digits (0-9)
    characters = string.ascii_letters + string.digits
    
    # Generate a list of random characters and join them into a string
    return ''.join(random.choices(characters, k=length))

class DatastoreUtilities():

    def __init__(self,config,project_name="worldbuilding",current_graph='lore'):

        # Temporary variables (Delete once fixed)
        self.arrow_scale = 0.9


        self.config = config
        self.kg_to_ds_path = config['kg_to_ds_map']
        self.project_name = project_name

        # Create the project folders in graphs and datasets if they do not exist.
        os.makedirs(f"{self.config['graph_dir']}/{self.project_name}", exist_ok=True)
        os.makedirs(f"{self.config['datastore_dir']}/{self.project_name}", exist_ok=True)
        
        self.current_graph = current_graph
        self.knowledge_graph_path = f"{self.config['graph_dir']}/{self.project_name}/{self.current_graph}_graph"
        self.datastore_path = f"{self.config['datastore_dir']}/{self.project_name}/{self.current_graph}_datastore"

        # Load knowledge-graph to datastore map

        if(not(os.path.exists(self.kg_to_ds_path))):
            # kg_to_ds_path should indicate the knowledge graph version contained in the current FAISS dataset and the ones not tracked
            kg2ds = dict()
            kg2ds[self.project_name] = dict()
            kg2ds[self.project_name][self.current_graph] = dict()
            kg2ds[self.project_name][self.current_graph]['datastore_last_saved'] = 0
            kg2ds[self.project_name][self.current_graph]['nodes_not_in_datastore'] = []
            kg2ds[self.project_name][self.current_graph]['nodes_not_in_graph'] = []
            self.kg2ds_map = kg2ds

            with open(config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)
        else:
            with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
                self.kg2ds_map = json.load(f)

            if(not(self.project_name) in self.kg2ds_map.keys()):
                self.kg2ds_map[self.project_name] = dict()

            if(not(self.current_graph) in self.kg2ds_map[self.project_name].keys()):
                self.kg2ds_map[self.project_name][self.current_graph] = dict()
                self.kg2ds_map[self.project_name][self.current_graph]['datastore_last_saved'] = 0
                self.kg2ds_map[self.project_name][self.current_graph]['nodes_not_in_datastore'] = []
                self.kg2ds_map[self.project_name][self.current_graph]['nodes_not_in_graph'] = []

        # Load up the current graph
        
        if(os.path.exists(f"{self.knowledge_graph_path}.gml")):
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_path}.gml")
        else:
            self.knowledge_graph = nx.MultiDiGraph()
            nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_path}.gml")

        print(f"Loaded Graph Version : {self.knowledge_graph_path} : {self.knowledge_graph}")
        
        # Load up the datastore for the current graph

        if(os.path.exists(f"{self.datastore_path}")):
            self.dataset = datasets.load_from_disk(f"{self.datastore_path}")
            self.dataset.load_faiss_index("embeddings", f"{self.datastore_path}.faiss")
        else:
            # No datastore was found - Create an empty dataset to populate.
            features = Features({
                "topic": Value("string"),
                "type": Value("string"),
                "node_id": Value("string"),
                "text": Value("string"),
            })
            self.dataset = Dataset.from_dict({
                                                "topic": [], 
                                                "type": [],
                                                "node_id":[],
                                                "text":[]
                                            }, 
                                            features=features)

        # Update the kg_to_ds_map
        with open(self.config['kg_to_ds_map'], "w") as f:
            json.dump(self.kg2ds_map, f, indent=4)

        # Find nodes based on aliases and map them back to the node
        self.reload_lookups()
     
    # FAISS DATASET FUNCTIONS

    def calculate_graph_datastore_diff(self,write_to_kg2ds=True):
        # Call this function after saving graphs and datasets.

        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        nodes_not_in_graph = [row['node_id'] for row in self.dataset if not(self.knowledge_graph.has_node(row['node_id']))]
        nodes_not_in_dataset = [node for node in self.knowledge_graph.nodes() if not(node in self.dataset['node_id'])]
        nodes_with_unsaved_changes = [node for node in self.knowledge_graph.nodes() if not(self.knowledge_graph.nodes[node]['updated']>self.kg2ds_map[self.project_name][self.current_graph]['datastore_last_saved'])]

        if(write_to_kg2ds):
            self.kg2ds_map[self.project_name][self.current_graph]['nodes_not_in_graph'] = nodes_not_in_graph
            self.kg2ds_map[self.project_name][self.current_graph]['nodes_not_in_datastore'] = nodes_not_in_dataset

            with open(self.config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

        return nodes_not_in_graph,nodes_not_in_dataset,nodes_with_unsaved_changes

    def add_alias(self,node_name, alias):
        node_id = self.get_node_id(node_name)

        assert self.get_node_layer(node_id)=='topic',f"{node_id} is not a topic! Aliases may only be assigned to topics!"

        node_layer = self.get_node_layer(node_id)
        node_type = self.knowledge_graph.nodes[node_id]['type']

        node_id = self.add_node(node_id,node_layer,type=node_type,aliases=[alias])

        self.save_graph()

        return self.knowledge_graph.nodes[node_id]['metadata']['aliases']

    def reload_lookups(self):
        topic_nodes = list(filter(lambda node:'aliases' in node[1]['metadata'].keys(),self.knowledge_graph.nodes(data=True)))

        # Lookups for Topic nodes
        self.all_aliases = sorted([alias for node in topic_nodes for alias in node[1]['metadata']['aliases']],key=len,reverse=True)
        self.alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias in self.all_aliases)})\b"
        self.node_finder = re.compile(self.alias_pattern, flags=re.IGNORECASE)
        self.node_lookup = {alias:node[0] for node in topic_nodes for alias in node[1]['metadata']['aliases']}
        self.inverse_lookup = {alias:node[1]['text'] for node in topic_nodes for alias in node[1]['metadata']['aliases']}

        # Initialize ahocorasic automaton for mention detection
        self.topic_finder = ahocorasick.Automaton()

        for topic in topic_nodes:
            for alias in self.knowledge_graph.nodes[topic[0]]['metadata']['aliases']:
                self.topic_finder.add_word(alias.lower(), (self.knowledge_graph.nodes[topic[0]]['text'], alias))
                
        # Compile the automaton for lightning-fast multi-pattern searching
        self.topic_finder.make_automaton()

    def get_alias_dict(self,text):
        nodes_in_text = [self.knowledge_graph.nodes[self.node_lookup[name]] for name in list(set(self.node_finder.findall(text))) if len(name)>0]

        alias_dict = dict()
        for node in nodes_in_text:
            alias_dict[node['text']] = dict()
            alias_dict[node['text']]['type'] = node['type']
            alias_dict[node['text']]['aliases'] = node['metadata']['aliases']

        return alias_dict

    def load_embedding_model(self):
        from sentence_transformers import SentenceTransformer
        self.text_embedding_model = SentenceTransformer(self.config['text_embedding_model'])

    def add_nodes_to_dataset(self,nodes_to_add):
        # Add a set of nodes to the dataset

        node_documents = []
        for node in nodes_to_add:
            node_documents.append({
                                'node_id':node[0],
                                'name':node[1]['text'],
                                'text':node[1]['metadata']['summary'],
                                'type':node[1]['type']
                            })

        graph_dataset = Dataset.from_list(node_documents)

        return graph_dataset

    def update_faiss_dataset(self):

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}

        self.save_graph()
        nodes_to_update = self.get_nodes_to_add_to_faiss()

        if(self.dataset):
            nodes_to_remove = [row['node_id'] for row in self.dataset if not(self.knowledge_graph.has_node(row['node_id']))]
        else:
            nodes_to_remove = []

        if(len(nodes_to_update)>0 or len(nodes_to_remove)>0):

            # Create a new dataset based on the recently updated nodes in the knowledge graph
            new_dataset = self.add_nodes_to_dataset(nodes_to_update)
            new_dataset = new_dataset.map(embed_text, batched=True, batch_size=8)
            print("NEW DATASET SIZE : ",len(new_dataset))
            for row in new_dataset:
                print(f"{row['node_id']} : {row['text']}")
            print()

            print("CURRENT DATASET SIZE : ",len(self.dataset))
            # Filter out data from existing dataset coming from the same nodes
            if "embeddings" in self.dataset.list_indexes():
                self.dataset.drop_index(index_name="embeddings")

            filtered_dataset = self.dataset
            for node in nodes_to_update:
                filtered_dataset = filtered_dataset.filter(lambda example: example["node_id"] != node[0])
            for node_id in nodes_to_remove:
                filtered_dataset = filtered_dataset.filter(lambda example: example["node_id"] != node_id)

            print("FILTERED DATASET SIZE : ",len(filtered_dataset))

            updated_dataset = concatenate_datasets([filtered_dataset,new_dataset])
            print("UPDATED DATASET SIZE : ",len(updated_dataset))

            embedding_dim = self.text_embedding_model.get_embedding_dimension()
            cosine_index = faiss.IndexFlatIP(embedding_dim)

            return updated_dataset, cosine_index

        else:

            return None, None

    def overwrite_faiss_dataset(self,new_dataset,cosine_index,temp_name='_temp'):
        
        if(new_dataset):
            temp_dataset_path = f"{self.config['datastore_dir']}/{self.project_name}/{temp_name}"
            temp_faiss_path = f"{self.config['datastore_dir']}/{self.project_name}/{temp_name}.faiss"

            new_dataset.info.description = str(int())

            # Save to temporary location
            new_dataset.save_to_disk(temp_dataset_path)
            new_dataset.add_faiss_index(column="embeddings", custom_index=cosine_index)

            if os.path.exists(temp_faiss_path):
                os.remove(temp_faiss_path)
            new_dataset.save_faiss_index("embeddings", temp_faiss_path)

            # Delete old dataset if exists
            for folder in os.listdir(f"{self.config['datastore_dir']}/{self.project_name}"):
                if(os.path.isdir(f"{self.config['datastore_dir']}/{self.project_name}/{folder}") and not(folder==temp_name)):
                    shutil.rmtree(f"{self.config['datastore_dir']}/{self.project_name}/{folder}")
                    os.remove(f"{self.config['datastore_dir']}/{self.project_name}/{folder}.faiss")

            # Rename temporary location
            os.rename(temp_dataset_path,f"{self.datastore_path}")
            os.rename(temp_faiss_path,f"{self.datastore_path}.faiss")

            self.kg2ds_map[self.project_name][self.current_graph]['datastore_last_saved'] = int(time.time())

            with open(self.kg_to_ds_path, "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

            # Add timestamp to nodes
            self.reload_faiss_dataset()

            print("FAISS datastore successfully created and saved!")
        else:
            print("No new data to be added!")

    def reload_faiss_dataset(self):

        # Reload kg_to_ds_map
        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        if(os.path.exists(f"{self.datastore_path}")):
            self.dataset = datasets.load_from_disk(f"{self.datastore_path}")

            if not("embeddings" in self.dataset.list_indexes()):
                embedding_dim = 1024
                print(embedding_dim)
                cosine_index = faiss.IndexFlatIP(embedding_dim)
                self.dataset.add_faiss_index(column="embeddings",custom_index=cosine_index)

            self.dataset.load_faiss_index("embeddings", f"{self.datastore_path}.faiss")
            self.calculate_graph_datastore_diff()
        else:
            self.dataset = None
        print(self.dataset)

    def filter_faiss_dataset(self,layer):

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}

        # Filter out data from existing dataset coming from the same nodes

        temp_dataset = self.dataset
        print(temp_dataset.list_indexes())
        if "embeddings" in temp_dataset.list_indexes():
            temp_dataset.drop_index("embeddings")

        filtered_dataset = temp_dataset.filter(lambda example: example["layer"] == layer)

        embedding_dim = self.text_embedding_model.get_embedding_dimension()
        cosine_index = faiss.IndexFlatIP(embedding_dim)
        filtered_dataset.add_faiss_index(column="embeddings", custom_index=cosine_index, index_name="filtered_embeddings_index")

        # Re-add the index to the original dataset
        #self.dataset.add_faiss_index(column="embeddings", custom_index=cosine_index)
        #self.dataset.load_faiss_index("embeddings", f"{self.datastore_path}.faiss")

        return filtered_dataset

    # EDGE FUNCTIONS

    def add_edge(self,head,tail,type,tag,text,**edge_metadata):
        # Add an edge to the graph
        # Keep edges as light as possible

        # Check if the labels and heads and tails match up
        #assert label in allowed_edge_labels, f"Label must be one of {allowed_edge_labels}, but got '{label}'"
        assert self.get_node_id(head) in self.knowledge_graph.nodes(), f"Head node '{head}' not found in knowledge graph!"
        assert self.get_node_id(tail) in self.knowledge_graph.nodes(), f"Tail node '{tail}' not found in knowledge graph!"
        #if(label=='relationship'): assert (self.get_node_layer(head)=='topic' and self.get_node_layer(tail)=='topic'), f"Node type mismatch for 'relationship' type edge! expected head='topic' and tail='topic'. Got head='{self.get_node_layer(head)}' and tail='{self.get_node_layer(tail)}'"
        #if(label=='involvement'): assert (self.get_node_layer(head)=='topic' and self.get_node_layer(tail)=='plot'), f"Node type mismatch for 'involvement' type edge! expected head='topic' and tail='plot'. Got head='{self.get_node_layer(head)}' and tail='{self.get_node_layer(tail)}'"
        #if(label=='description'): assert (self.get_node_layer(head)=='topic' and self.get_node_layer(tail)=='lore'), f"Node type mismatch for 'description' type edge! expected head='topic' and tail='lore'. Got head='{self.get_node_layer(head)}' and tail='{self.get_node_layer(tail)}'"
        #if(label=='scene'): assert (self.get_node_layer(head)=='plot' and self.get_node_layer(tail)=='plot'), f"Node type mismatch for 'scene' type edge! expected head='plot' and tail='plot'. Got head='{self.get_node_layer(head)}' and tail='{self.get_node_layer(tail)}'"
        #if(label=='info'): assert (self.get_node_layer(head)=='topic' and self.get_node_layer(tail)=='fact'), f"Node type mismatch for 'info' type edge! expected head='topic' and tail='fact'. Got head='{self.get_node_layer(head)}' and tail='{self.get_node_layer(tail)}'"

        time_now = int(time.time())

        metadata = dict()
        if(len(edge_metadata.keys())>0):
            for key in edge_metadata.keys():
                metadata[key] = edge_metadata[key]

        # Check if similar edge already exists.
        key = self.knowledge_graph.add_edge(self.get_node_id(head),
                                            self.get_node_id(tail),
                                            type=type,
                                            tags=[tag],
                                            text=text,
                                            updated=time_now,
                                            metadata=metadata)

        print(f"ADDED NEW EDGE BETWEEN '{self.get_node_name(head)}' and '{self.get_node_name(tail)}' : {self.knowledge_graph.edges[self.get_node_id(head),self.get_node_id(tail),key]}")

        return (self.get_node_id(head),self.get_node_id(tail),key)

    def remove_edge(self,head,tail,type=None,tag=None):
        head_id = self.get_node_id(head)
        tail_id = self.get_node_id(tail)


        if(self.knowledge_graph.has_edge(head_id,tail_id)):
            edge_data = self.knowledge_graph.get_edge_data(head_id,tail_id)
            removed_edges = dict()

            if(type):
                if(tag):
                    # Remove all edges with type and tag
                    keys_to_remove = [key for key in edge_data.keys() if edge_data[key]['type']==type and edge_data[key]['tag']==tag]
                else:
                    # Remove all edges with type
                    keys_to_remove = [key for key in edge_data.keys() if edge_data[key]['type']==type]
            else:
                # Remove all edges
                keys_to_remove = [key for key in edge_data.keys()]

            print(f"Removing edges between {self.get_node_name(head_id)} and {self.get_node_name(tail_id)} with keys {keys_to_remove}")
            for key in keys_to_remove:
                removed_edges[key] = (head_id,tail_id,self.knowledge_graph.edges[head_id,tail_id,key]['text'])
                self.knowledge_graph.remove_edge(head_id,tail_id,key)

            return removed_edges
        else:
            return dict()

    def get_edges_of_type(self,type,tag=None):

        for edge in self.knowledge_graph.edges(keys=True):
            print(edge)
            print(self.knowledge_graph.edges[edge[0],edge[1],edge[2]])

        if(tag):
            # Check tag similarity later if needed.
            return [(head, tail, key) for head,tail,key in self.knowledge_graph.edges(keys=True) if (self.knowledge_graph.edges[head,tail,key]['type']==type and tag in self.knowledge_graph.edges[head,tail,key]['tags'])]
        else:
            return [(head, tail, key) for head,tail,key in self.knowledge_graph.edges(keys=True) if (self.knowledge_graph.edges[head,tail,key]['type']==type)]

    def check_edges_for_redundancy(self,head,tail,new_edge_text,threshold=0.9):
        head = self.get_node_id(head)
        tail = self.get_node_id(tail)

        edge_texts = []
        if(self.knowledge_graph.get_edge_data(head,tail)):
            for key,edge_info in self.knowledge_graph.get_edge_data(head,tail).items():
                edge_texts.append(edge_info['text'])

            cosine_sims = self.get_cosine_similarity([new_edge_text],edge_texts)[0]

            for idx in range(len(edge_texts)):
                print(f"{cosine_sims[idx]} --> {edge_texts[idx]}")
            
            return (torch.any(cosine_sims>threshold))
        else:
            return False

    # NODE FUNCTIONS

    # Create and Delete functions
    def add_node(self,node_name,type,tag,**metadata):
        # NOTE : Aliases needed for relevant topic retrieval
        
        # Check if the node exists in the graph (Mostly useful for `topic` nodes)

        node_id = None
        if(not(self.get_node_id(node_name))):

            node_name,_ = self.resolve_aliases(node_name)

            # Create a new node if knowledge graph does not exist
            node_id = get_code(node_name)

            self.knowledge_graph.add_node(node_id,
                                        updated=int(time.time()),
                                        text=node_name,
                                        type=type,
                                        tags=[tag],
                                        metadata=dict())

            if(metadata):
                pass
            else:
                metadata = dict()        

            if('aliases' in metadata.keys()):
                metadata['aliases'].append(node_name)
                metadata['aliases'] = list(set(metadata['aliases']))
            else:
                metadata['aliases'] = node_name
                
            if('summary' in metadata.keys()):
                metadata['summary_updated'] = int(time.time())
            else:
                metadata['summary'] = ""
                metadata['summary_updated'] = 0


            self.add_metadata(node_id,replace=True,**metadata)

            print(f"Added node [{node_id}] {self.get_node_name(node_id)} --> {type} : {tag}!")
        else:
            node_id = self.get_node_id(node_name)
            self.knowledge_graph.nodes[node_id]['updated'] = int(time.time())
            self.knowledge_graph.nodes[node_id]['type'] = type
            self.knowledge_graph.nodes[node_id]['tags'].append(tag)
            self.knowledge_graph.nodes[node_id]['tags'] = list(set(self.knowledge_graph.nodes[node_id]['tags']))
            self.add_metadata(node_id,replace=False,**metadata)
            print(f"Node already exists! Updated Node : {self.knowledge_graph.nodes[node_id]}")



        return node_id
                
    def remove_node(self,node):
        # Cleanly remove a node from the knowledge graph
        node_id = self.get_node_id(node)
        
        if(node_id):
            node_data = self.knowledge_graph.nodes[node_id]

            edges_to_remove = []
            for edge in self.knowledge_graph.out_edges([node_id],data=True):
                edges_to_remove.append((edge[0],edge[1],edge[2]))
            for edge in self.knowledge_graph.in_edges([node_id],data=True):
                edges_to_remove.append((edge[0],edge[1],edge[2]))

            for head,tail,key in edges_to_remove:
                # If head or tail is different, create a new edge that loops back to the head or tail that is not the node to be removed.

                edge_type       = self.knowledge_graph.edges[head,tail,key]['type']
                edge_tag    = self.knowledge_graph.edges[head,tail,key]['tag']
                edge_text       = self.knowledge_graph.edges[head,tail,key]['text']
                if(not(head==node_id)):
                    self.add_edge(head,head,edge_type,edge_tag,edge_text)
                if(not(tail==node_id)):
                    self.add_edge(tail,tail,edge_type,edge_tag,edge_text)

                self.knowledge_graph.remove_edge(head,tail,key)
                print(f"Removed edge between {head} and {tail} with key {key} : {self.knowledge_graph.edges[head,tail,key]['text']}")

            print(f"Removing node [{node_id}] ({self.get_node_name(node_id)})")
            self.knowledge_graph.remove_node(node_id)

            return node_data
        else:
            print(f"No node similar to {node} found!")
            return None

    def add_metadata(self,node,replace=True,**metadata):

        node_id = self.get_node_id(node)

        if(node_id):
            if(replace):
                for key in metadata.keys():
                    self.knowledge_graph.nodes[node_id]['metadata'][key] = metadata[key]
                    print(f"Replaced Metadata : {key} = {self.knowledge_graph.nodes[node_id]['metadata'][key]}")

                self.knowledge_graph.nodes[node_id]['updated'] = int(time.time())

                if('summary' in metadata.keys()):
                    self.knowledge_graph.nodes[node_id]['metadata']['summary_updated'] = int(time.time())
            else:
                # Make separate functions if metadata is a list or a dict

                # If list
                for key in metadata.keys():
                    if(key in self.knowledge_graph.nodes[node_id]['metadata'].keys()):
                        self.knowledge_graph.nodes[node_id]['metadata'][key] += metadata[key]
                        self.knowledge_graph.nodes[node_id]['metadata'][key] = list(set(self.knowledge_graph.nodes[node_id]['metadata'][key]))
                    else:
                        self.knowledge_graph.nodes[node_id]['metadata'][key] = metadata[key]
                    print(f"Added Metadata : {key} = {self.knowledge_graph.nodes[node_id]['metadata'][key]}")

                self.knowledge_graph.nodes[node_id]['updated'] = int(time.time())

    # Read functions
    def get_node_name(self,node_id):
        if(node_id in self.knowledge_graph.nodes()):
            return self.knowledge_graph.nodes[node_id]['text']
        elif(get_code(node_id) in self.knowledge_graph.nodes()):
            return self.knowledge_graph.nodes[get_code(node_id)]['text']
        else:
            return None

    def get_node_id(self,node_name):

        if(get_code(node_name) in self.knowledge_graph.nodes()):
            return get_code(node_name)
        elif(node_name in self.knowledge_graph.nodes()):
            return node_name
        elif(node_name in self.node_lookup.keys()):
            return self.node_lookup[node_name]
        else:
            return None

    def get_node_type(self,node_name):

        node_id = self.get_node_id(node_name)
        if(node_id):
            return self.knowledge_graph.nodes[node_id]['type']
        else:
            return None

    def get_node_tags(self,node_name):

        node_id = self.get_node_id(node_name)
        return self.knowledge_graph.nodes[node_id]['tags']

    def get_node_aliases(self,node_name):

        node_id = self.get_node_id(node_name)

        if('aliases' in self.knowledge_graph.nodes[node_id]['metadata'].keys()):
            return self.knowledge_graph.nodes[node_id]['metadata']['aliases']
        else:
            return []

    def get_nodes_of_type(self,type,tag=None):

        if(tag):
            # Check tag similarity later if needed.
            return [node for node in self.knowledge_graph.nodes() if (self.knowledge_graph.nodes[node]['type']==type and tag in self.knowledge_graph.nodes[node]['tags'])]
        else:
            return [node for node in self.knowledge_graph.nodes() if self.knowledge_graph.nodes[node]['type']==type]

    def get_all_neighbors(self,node_name):

        node_id = self.get_node_id(node_name)

        all_neighbors = dict()
        all_neighbors['predecessors'] = list(self.knowledge_graph.predecessors(node_id))
        all_neighbors['successors'] = list(self.knowledge_graph.successors(node_id))
        all_neighbors['all'] = list(set(list(self.knowledge_graph.successors(node_id))+list(self.knowledge_graph.predecessors(node_id))))

        return all_neighbors

    def get_all_edges(self,node_name):
        node_id = self.get_node_id(node_name)

        all_neighbors = self.get_all_neighbors(node_id)
        # Load predecessors and successors individually

        all_edges = []
        for neighbor in all_neighbors['successors']:
            edge_dict = self.knowledge_graph.get_edge_data(node_id,neighbor)
            all_edges += [(key,edge_dict[key]) for key in edge_dict.keys()]

        for neighbor in all_neighbors['predecessors']:
            edge_dict = self.knowledge_graph.get_edge_data(neighbor,node_id)
            all_edges += [(key,edge_dict[key]) for key in edge_dict.keys()]

        return all_edges

    def get_all_edges_of_type(self,node_name,allowed_edge_types):
        node_id = self.get_node_id(node_name)

        all_edges = dict()
        if(node_id):
            all_neighbors = self.get_all_neighbors(node_id)
            # Load predecessors and successors individually

            for neighbor in all_neighbors['all']:
                all_edges[self.get_node_name(neighbor)] = []

            for neighbor in all_neighbors['successors']:
                edge_dict = self.knowledge_graph.get_edge_data(node_id,neighbor)
                all_edges[self.get_node_name(neighbor)] += [(key,edge_dict[key]) for key in edge_dict.keys() if edge_dict[key]['type'] in allowed_edge_types]

            for neighbor in all_neighbors['predecessors']:
                edge_dict = self.knowledge_graph.get_edge_data(neighbor,node_id)
                all_edges[self.get_node_name(neighbor)] += [(key,edge_dict[key]) for key in edge_dict.keys() if edge_dict[key]['type'] in allowed_edge_types]

        return all_edges

    def get_mentioned_topic(self,text):

        mentioned_topics = [name for name in list(set(self.node_finder.findall(text))) if len(name)>0]

        return self.get_node_id(mentioned_topics[0])

    # Old functions

    def get_relevant_topics(self,text,threshold=0.8,all=True):

        # Based on text, find the most similar node and return.

        mentioned_topics = [self.get_node_id(name) for name in list(set(self.node_finder.findall(text))) if len(name)>0]

        if(len(mentioned_topics)>0):
            if(all==False):
                return mentioned_topics[0]

        query,_ = self.resolve_aliases(text)
        query_vector = self.embed_text(query)

        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=20)

        for idx,score in enumerate(scores):
            #print(f"{examples['text'][idx]} : {score}")
            if(score>threshold):
                mentioned_topics.append(examples['node_id'][idx])

        if(all==False):
            return mentioned_topics[0]
        else:
            return list(set(mentioned_topics))

    def get_nodes_to_add_to_faiss(self):
        # Return a list of nodes that have been updated, but not added to the FAISS dataset
        # Decompose a node into wiki keys and check if the ids are in the dataset.

        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        # Get only the non-topic nodes that have been updated after the last datastore save.
        return [node for node in self.knowledge_graph.nodes(data=True) if (node[1]['updated']>self.kg2ds_map[self.project_name][self.current_graph]['datastore_last_saved'])]

    def get_adjacent_nodes(self,node):
        # For a given node, get all neighbors irrespective of direction
        node_id = self.get_node_id(node)
        return self.knowledge_graph.successors(node_id)|self.knowledge_graph.predecessors(node_id)

    # GRAPH FUNCTIONS

    def find_topic_mentions(self, text):
        
        matches = []
        text_lower = text.lower()
        text_len = len(text)
        
        # 2. Scan the text chunk in a single pass
        for end_index, (canonical_name, matched_alias) in self.topic_finder.iter(text_lower):
            start_index = end_index - len(matched_alias) + 1
            
            # 3. Apply a Word Boundary Check (prevents sub-string collisions like "Cat" in "Catalog")
            is_start_boundary = (start_index == 0) or (not text[start_index - 1].isalnum())
            is_end_boundary = (end_index + 1 == text_len) or (not text[end_index + 1].isalnum())
            
            if not (is_start_boundary and is_end_boundary):
                continue  # Skip partial word matches
                
            # 4. Record the match, extracting the exact casing from the original text chunk
            '''
            matches.append({
                "canonical_name": canonical_name,
                "matched_text": text[start_index:end_index + 1],
                "start_offset": start_index,
                "end_offset": end_index + 1
            })
            '''
            matches.append(text[start_index:end_index + 1])
            
        return list(set(matches))

    def extrapolate_for_node(self,node_id,mode):

        if(not node_id in self.knowledge_graph.nodes()):
            node_id = self.get_node_id(node_id)

        node = self.knowledge_graph.nodes[node_id]

        mentioned_topics = [name for name in list(set(self.node_finder.findall(node['text']))) if len(name)>0]

        new_edge = None
        for topic in mentioned_topics:
            if(mode=='info'):
                new_edge = self.add_edge(self.node_lookup[topic],node_id,'info')
            if(mode=='description'):
                new_edge = self.add_edge(self.node_lookup[topic],node_id,'description')
            elif(mode=='involvement'):
                new_edge = self.add_edge(self.node_lookup[topic],node_id,'involvement')
            elif(mode=='relationship'):

                new_edge = None
                # Get 2 hop neighbors for topic nodes
                assert self.get_node_layer(node_id) == 'topic'

                neighbors = self.get_hop_neighbors([node_id])

                # Keep all the non-topic nodes for the first hop
                filtered_neighbors = [n for n in neighbors if not(self.get_node_layer(n)=='topic')]
                print([self.get_node_name(n) for n in filtered_neighbors])

                # Keep all the topic nodes for the second hop
                topic_neighbors = self.get_hop_neighbors(filtered_neighbors)
                topic_neighbors = [n for n in topic_neighbors if not(n==node_id) and self.get_node_layer(n)=='topic'] 
                print([self.get_node_name(n) for n in topic_neighbors])

                
                for n_topic in topic_neighbors:
                    new_edge = self.add_edge(node_id,n_topic,'relationship')

        return new_edge

    def order_scenes(self,ordered_nodes):

        def get_first_common_element(list_a, list_b):
            # Convert list_b to a set for O(1) lookups
            set_b = set(list_b)
            
            # Iterate through list_a to preserve the original order
            for item in list_a:
                if item in set_b:
                    return item
                    
            return None  # Return None if no common element exists

        assert all([self.knowledge_graph.nodes[node]['layer']=='plot' for node in ordered_nodes]), f"All nodes are not plot nodes!"

        added_edges = []
        edges_to_reconsider = dict()

        edges_to_reconsider['least_cut'] = []
        edges_to_reconsider['full_overlap'] = []

        plot_graph = self.filter_graph(filter_layer='plot')
        for idx in range(len(ordered_nodes)-1):
            head = ordered_nodes[idx]
            tail = ordered_nodes[idx+1]

            chapter_overlap, heads_overlap, tails_overlap = self.do_chapters_overlap(head,tail)
            if chapter_overlap:
                full_overlap_dict = dict()
                print(f"Adding a path between {head} and {tail} will form a cycle!")
                # One of the predecessors or successors of the head and tail must be common.
                # If we make a cut at the common node within the paths to the common heads or tails, we create new branches.

                if(heads_overlap and tails_overlap):
                    # If heads AND tails overlap, it means that a direct path already exists between the two nodes
                    print(f"{head} and {tail} exist on the same path! Don't create a shortcut!")

                    # Find a path between the two nodes, and delete any edge between them.
                    common_heads = set(self.get_chapter_heads(head))&set(self.get_chapter_heads(tail))
                    common_tails = set(self.get_chapter_tails(head))&set(self.get_chapter_tails(tail))

                    common_chapter_paths = []
                    for c_head in common_heads:
                        for c_tail in common_tails:
                            nodes_in_path = nx.shortest_path(self.knowledge_graph,c_head,c_tail)
                            common_chapter_paths += [(u, v) for u, v in nx.utils.pairwise(nodes_in_path) if self.knowledge_graph.has_edge(u, v)]

                    common_chapter_paths = list(set(common_chapter_paths))
                    common_chapter_nodes = list(set([item for tup in common_chapter_paths for item in tup]))

                    print("Full Overlap")
                    print(common_chapter_paths)
                    print(common_chapter_nodes)

                    full_overlap_dict['chapter_edges'] = common_chapter_paths
                    full_overlap_dict['chapter_nodes'] = common_chapter_nodes

                    node_path_nodes = nx.shortest_path(self.knowledge_graph,head,tail)
                    node_path_edges = [(u, v) for u, v in nx.utils.pairwise(node_path_nodes) if self.knowledge_graph.has_edge(u, v)]

                    full_overlap_dict['common_edges'] = node_path_edges
                    full_overlap_dict['common_nodes'] = node_path_nodes

                    print(node_path_edges)
                    print(node_path_nodes)

                    edges_to_reconsider['full_overlap'].append(full_overlap_dict)


                elif(heads_overlap):

                    partial_overlap_dict = dict()

                    # Find the common heads for the two nodes
                    common_heads = set(self.get_chapter_heads(head))&set(self.get_chapter_heads(tail))

                    # For each path from each head, find the common path node for both nodes
                    for chapter_head in common_heads:
                        head_path1 = nx.shortest_path(self.knowledge_graph,chapter_head,head)
                        head_path2 = nx.shortest_path(self.knowledge_graph,chapter_head,tail)

                        head_path1.reverse()
                        head_path2.reverse()

                        last_common_predecessor = get_first_common_element(head_path1,head_path2)

                        head_path1.reverse()
                        head_path2.reverse()

                        # Least Cut Edge format for head : (node1, node2), (common_predecessor,node_after_that1), (common_predecessor,node_after_that2)
                        least_cut_for_node1 = head_path1[head_path1.index(last_common_predecessor)+1]
                        least_cut_for_node2 = head_path2[head_path2.index(last_common_predecessor)+1]

                        partial_overlap_dict['new_edge'] = (head,tail)
                        partial_overlap_dict['option_1'] = (last_common_predecessor,least_cut_for_node1)
                        partial_overlap_dict['option_2'] = (last_common_predecessor,least_cut_for_node2)
                        partial_overlap_dict['nodes_in_path'] = list(set(head_path1+head_path2))

                        edges_to_reconsider['least_cut'].append(partial_overlap_dict)

                    print(f"Common Predecessor is {last_common_predecessor}")

                elif(tails_overlap):
                    
                    # Find the common tails for the two nodes
                    common_tails = set(self.get_chapter_tails(head))&set(self.get_chapter_tails(tail))

                    # For each path from each head, find the common path node for both nodes
                    for chapter_tail in common_tails:
                        partial_overlap_dict = dict()
                        tail_path1 = nx.shortest_path(self.knowledge_graph,head,chapter_tail)
                        tail_path2 = nx.shortest_path(self.knowledge_graph,tail,chapter_tail)

                        first_common_successor = get_first_common_element(tail_path1,tail_path2)

                        # Least Cut Edge format for head : (node1, node2), (node_before_that1,common_successor), (node_before_that2,common_successor)
                        least_cut_for_node1 = tail_path1[tail_path1.index(first_common_successor)-1]
                        least_cut_for_node2 = tail_path2[tail_path2.index(first_common_successor)-1]

                        partial_overlap_dict['new_edge'] = (head,tail)
                        partial_overlap_dict['option_1'] = (least_cut_for_node1,first_common_successor)
                        partial_overlap_dict['option_2'] = (least_cut_for_node2,first_common_successor)
                        partial_overlap_dict['nodes_in_path'] = list(set(tail_path1+tail_path2))

                        #least_cut_edge = ((head,tail),(least_cut_for_node1,first_common_successor),(least_cut_for_node2,first_common_successor),tail_path1,tail_path2)
                        edges_to_reconsider['least_cut'].append(partial_overlap_dict)

                    print(f"Common Successor is {first_common_successor}")
            else:
                print(f"Added a path between {head} and {tail}!")
                new_edge = self.add_edge(head,tail,'scene')
                if(new_edge):
                    added_edges.append(new_edge)

        return added_edges, edges_to_reconsider

    def get_chapter_heads(self,node):
        
        node_id = self.get_node_id(node)
        assert self.get_node_layer(node_id)=='plot'

        nodes = [node_id]
        heads = []
        for node in nodes:
            current_node = node

            while(True):
                # Check if the current node has any plot predecessors
                previous_plots = [p for p in self.knowledge_graph.predecessors(current_node) if self.get_node_layer(p)=='plot']

                if(len(previous_plots)==0):
                    # This is the head of the chapter. Move to the next node
                    heads.append(current_node)
                    break
                elif(len(previous_plots)==1):
                    # This is the only previous plot. Set this as the current node
                    current_node = previous_plots[0]
                else:
                    # The current node is a fork. Choose one of the branches and append the others to nodes.
                    current_node = previous_plots[0]
                    nodes += previous_plots[1:]

        return heads

    def get_chapter_tails(self,node):

        node_id = self.get_node_id(node)
        assert self.get_node_layer(node_id)=='plot'

        nodes = [node_id]
        tails = []
        for node in nodes:
            current_node = node

            while(True):
                # Check if the current node has any plot successors
                next_plots = [p for p in self.knowledge_graph.successors(current_node) if self.get_node_layer(p)=='plot']

                if(len(next_plots)==0):
                    # This is the tail of the chapter. Move to the next node
                    tails.append(current_node)
                    break
                elif(len(next_plots)==1):
                    # This is the only next plot. Set this as the current node
                    current_node = next_plots[0]
                else:
                    # The current node is a fork. Choose one of the branches and append the others to nodes.
                    current_node = next_plots[0]
                    nodes += next_plots[1:]

        return tails

    def do_chapters_overlap(self,node1,node2):

        node1_id = self.get_node_id(node1)
        node2_id = self.get_node_id(node2)

        heads_overlap = not set(self.get_chapter_heads(node1_id)).isdisjoint(self.get_chapter_heads(node2_id))
        tails_overlap = not set(self.get_chapter_tails(node1_id)).isdisjoint(self.get_chapter_tails(node2_id))

        chapter_overlap = heads_overlap or tails_overlap

        return (chapter_overlap, heads_overlap, tails_overlap)

    def sequence_scenes(self,before,after):

        before_id = self.find_similar_nodes(before,k=1,filter='plot')[0]
        after_id = self.find_similar_nodes(after,k=1,filter='plot')[0]

        #print(f"Before - {before_text} : {before_score}")
        #print(f"After - {after_text} : {after_score}")
        
        edge = self.add_edge(before_id,after_id,'scene')

        return edge

    def filter_graph(self,filter_layer):

        def filter_function(u):
            return self.knowledge_graph.nodes[u]["layer"] == filter_layer

        filtered_graph = nx.subgraph_view(self.knowledge_graph, filter_node=filter_function)

        return filtered_graph

    def find_path_between(self,source,target,layer):
        # Returns a sequence of `plot` nodes

        filtered_graph = self.filter_graph(layer)

        #source_id = self.find_similar_nodes(source,k=1,filter=layer)[0]
        #target_id = self.find_similar_nodes(target,k=1,filter=layer)[0]

        assert source in filtered_graph.nodes()
        assert target in filtered_graph.nodes()

        #print(f"Source : {source_text} - ({source_score})")
        #print(f"Target : {target_text} - ({target_score})")

        if nx.has_path(filtered_graph, source, target):
            path = nx.shortest_path(filtered_graph, source=source, target=target)
        else:
            path = None

        return path

    def propagate_plot_nodes(self,node,n_hops=2,direction='forward'):
        # Returns the next plot nodes from a given node
        assert self.get_node_layer(node)=='plot',f"{node} is not a valid plot node!"
        assert direction in ['forward','backward'], f"direction = {direction} must be one of 'forward' or 'backward'"

        node_id = self.get_node_id(node)

        adjacent_nodes = []

        for hop in range(n_hops+1):

            if(direction=='forward'):
                next_nodes = [tail for head,tail in self.knowledge_graph.out_edges(node_id) if self.get_node_layer(tail)=='plot']
            elif(direction=='backward'):
                next_nodes = [head for head,tail in self.knowledge_graph.in_edges(node_id) if self.get_node_layer(head)=='plot']

            old_node_id = node_id
            if(len(next_nodes)>0):
                node_id = next_nodes[0]
                adjacent_nodes.append(node_id)

            if(node_id==old_node_id):
                break

        if(direction=='backward'):
            adjacent_nodes.reverse()

        return adjacent_nodes

    def get_related_topics(self,node):
        # Given any node, get the list of neighboring topic nodes

        node_id = self.get_node_id(node)

        if(self.knowledge_graph.nodes[node_id]['layer']=='topic'):
            return [node_id]
        else:
            # `plot` or `lore` are successors of `topic` (i.e `topic` ---> `lore`)
            neighbors = self.knowledge_graph.predecessors(node_id)
            return [neighbor_id for neighbor_id in neighbors if self.knowledge_graph.nodes[neighbor_id]['layer']=='topic']

    def explore_neighborhood(self,node,n_hops=1,layer='topic'):
        # Returns a set of `topic` nodes

        filtered_graph = self.filter_graph(layer)

        new_neighborhood = [self.get_node_id(node)]

        for hop in range(n_hops):

            old_neighborhood = new_neighborhood 
            new_neighborhood = []
            for new_node_id in old_neighborhood:
                new_neighborhood += list(filtered_graph.neighbors(new_node_id))

            new_neighborhood = list(set(old_neighborhood+new_neighborhood))

        return new_neighborhood

    def get_hop_neighbors(self,nodes):

        neighbors = []
        for node in nodes:
            node_id = self.get_node_id(node)
            neighbors += set(self.knowledge_graph.successors(node_id))|set(self.knowledge_graph.predecessors(node_id))

        return [n for n in set(neighbors) if not(n in nodes)]

    def collect_lore_nodes(self,node):

        node_id = self.get_node_id(node)

        lore_nodes = []
        for head,tail,edge_info in self.knowledge_graph.edges(node_id,data=True):
            if edge_info['label']=='description':
                lore_nodes.append(tail)

        return lore_nodes

    def collect_fact_nodes(self,node,chosen_layer):

        node_id = self.get_node_id(node)

        fact_nodes = []
        for head,tail,edge_info in self.knowledge_graph.edges(node_id,data=True):
            #if (edge_info['label']=='info' and self.knowledge_graph.nodes[tail]['type']==chosen_layer):
            if (edge_info['label']=='info'):
                fact_nodes.append(tail)

        return fact_nodes

    def get_lore_from_nodes(self,nodes):

        lore_nodes = []
        for node in nodes:
            lore_nodes += self.collect_lore_nodes(node)
            
        lore_nodes = list(set(lore_nodes))

        lore = [self.knowledge_graph.nodes[lore_node]['text'] for lore_node in lore_nodes]

        for lore_info in lore:
            print(lore_info)

        return lore

    def cluster_lore(self,node):

        umap_reducer = umap.UMAP(
                            n_neighbors=4,        # Adjust based on dataset size (smaller = local, larger = global)
                            n_components=4,       # Target dimensions
                            metric='cosine',      # Cosine metric is ideal for text embeddings
                            random_state=42
                        )

        # Get the closest `topic` node
        node_id = self.get_node_id(node)

        if(not(self.get_node_layer(node_id)=='topic')):
            node_id = self.get_related_topics(node_id)[0]


        # Expand existing `lore` nodes --> old `facts`
        lore_nodes = self.collect_lore_nodes(node_id)
        old_facts = []
        for lore_node in lore_nodes:
            old_facts += self.get_node_metadata(lore_node,'facts')

        # Collect new `facts`
        new_facts = self.collect_fact_nodes(node_id,'lore')

        # Generate `fact` embeddings 
        all_facts_text = old_facts + [self.get_node_name(fact_node) for fact_node in new_facts]

        output = []
        if(len(all_facts_text)>5):
            print(all_facts_text)
            embeddings = self.embed_text(all_facts_text) # This may not work.
            #embeddings = torch.stack([torch.tensor(self.dataset['embeddings'][idx]) for idx in rows])
            print(embeddings.shape)

            reduced_embeddings = umap_reducer.fit_transform(embeddings)

            k = 3  # Number of neighbors per node
            nn = NearestNeighbors(n_neighbors=k, metric='cosine')
            nn.fit(reduced_embeddings)
            adjacency_matrix = nn.kneighbors_graph(reduced_embeddings).toarray()

            # 5. Convert adjacency matrix to a NetworkX Graph
            graph = nx.from_numpy_array(adjacency_matrix)

            # 6. Apply Graph-Based Community Detection (Louvain Algorithm)
            # This partitions the graph into tightly knit communities (clusters)
            communities = nx.community.louvain_communities(graph, seed=42)

            # 7. Map clusters back to the original texts
            for cluster_id, community in enumerate(communities):
                output.append([all_facts_text[node_idx] for node_idx in community])

        return output

    # Old functions

    def find_path(self,source,target,filtered_graph=None):

        source = self.get_node_id(source)
        target = self.get_node_id(target)

        if(not(filtered_graph)):
            filtered_graph = self.knowledge_graph

        path = None
        if(filtered_graph.has_node(source) and filtered_graph.has_node(target)):
            if nx.has_path(filtered_graph, source, target):
                path = nx.shortest_path(filtered_graph, source=source, target=target)

        return path

    def check_subpath(self,path1,path2):

        if(path1 and path2):
            if len(path1) >= len(path2):
                main_path = path1
                sub_path = path2
            else:
                main_path = path2
                sub_path = path1

            n = len(sub_path)

            paths_regular = any(sub_path == main_path[i : i + n] for i in range(len(main_path) - n + 1))

            reversed_subpath = sub_path[::-1]
            paths_inverted = any(reversed_subpath == main_path[i : i + n] for i in range(len(main_path) - n + 1))
            return any([paths_regular,paths_inverted])
        else:
            return False

    def export_graph(self):

        # Export all node and edge info

        graph_data = dict()

        edges= []
        for edge in self.knowledge_graph.edges(data=True,keys=True):
            edge_data = dict()
            edge_data['head'] = edge[0]
            edge_data['tail'] = edge[1]
            edge_data['key'] = edge[2]
            edge_data['text'] = edge[3]['text']
            edges.append(edge_data)
        
        nodes = dict()
        for node in self.knowledge_graph.nodes(data=True):
            
            nodes[node[0]] = dict()
            nodes[node[0]]['name'] = node[1]['name']
            nodes[node[0]]['layer'] = node[1]['layer']

        graph_data['nodes'] = nodes
        graph_data['edges'] = edges

        with open(f"{self.config['data_dir']}/graph_json_{self.kg2ds_map['unsaved_versions'][-1]}.json", "w") as f:
            json.dump(graph_data, f, indent=4)

    def save_graph(self,draw_figure=False):

        print("Saving Graph!")
        # Clean up old graphs
        # 1 day = 86400 seconds
        self.knowledge_graph.remove_edges_from(nx.selfloop_edges(self.knowledge_graph))
        nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_path}.gml")

        self.reload_graph()

    def reload_graph(self,most_recent=True):

        # Reload kg_to_ds_map
        
        if(os.path.exists(f"{self.knowledge_graph_path}.gml")):
            # load the latest knowledge graph
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_path}.gml")
            print(f"Reloaded Graph Version : {self.knowledge_graph_path} : {self.knowledge_graph}")
            self.reload_lookups()
            self.calculate_graph_datastore_diff()

            return True
        else:
            return False

    # RAG FUNCTIONS

    def resolve_aliases(self,text):

        # List to store our audit trail of replacements
        entity_dict = dict()

        def replacer(match):
            original = match.group(0)

            if(len(original.strip())>0):
                replacement = self.get_node_name(self.node_lookup[original])
                entity_dict[replacement]    = dict()
                entity_dict[replacement]['aliases']      = []
                entity_dict[replacement]['type']         = self.knowledge_graph.nodes[self.node_lookup[original]]['type']
                return replacement
            else:
                return original
            
        modified_text = re.sub(self.alias_pattern, replacer, text)

        return modified_text, entity_dict

    def find_similar_nodes(self,query,threshold=0.5,k=20,filter=None):
        
        dataset = self.dataset

        query,_ = self.resolve_aliases(query)
        query_vector = self.embed_text(query)

        scores, examples = dataset.get_nearest_examples("embeddings", query_vector, k=k)

        similar_nodes = []
        logging_info = []
        for idx,score in enumerate(scores):
            #print(f"{examples['text'][idx]} : {score}")
            if(filter):
                if(score>threshold and examples['layer'][idx]==filter):
                    logging_info.append((examples['node_id'][idx],score,examples['text'][idx]))
                    similar_nodes.append(examples['node_id'][idx])
            else:
                if(score>threshold):
                    logging_info.append((examples['node_id'][idx],score,examples['text'][idx]))
                    similar_nodes.append(examples['node_id'][idx])

        similar_nodes += [self.get_node_id(match) for match in self.find_topic_mentions(query) if match]

        print('similar_nodes : ',similar_nodes)

        return similar_nodes

    def find_nodes_in_clauses(self,clauses,threshold=0.6,k=20,filter=None):
        nodes_in_results = []
        for query in clauses:
            nodes_in_results += self.find_similar_nodes(query,threshold=threshold,k=k,filter=filter)

        return nodes_in_results

    def get_context_from_paths(self,paths,add_code=False,context_label='context'):

        if(add_code):
            for idx in range(len(paths)):
                random.shuffle(paths[idx])
            path_info = {str(idx):[f"[{node[0].capitalize()}{node[-1]}] : {self.knowledge_graph.nodes[node]['text']}" for node in path] for idx,path in enumerate(paths)}
        else:
            path_info = {str(idx):[self.knowledge_graph.nodes[node]['text'] for node in path] for idx,path in enumerate(paths)}

        all_path_nodes = []
        all_path_edges = []
        for path in paths:
            all_path_nodes += path

            for idx in range(len(path)-1):
                all_path_edges.append((path[idx],path[idx+1]))

        formatted_context = self.format_context(path_info,"path",add_code=add_code,label=context_label)

        return formatted_context, list(set(all_path_nodes)), list(set(all_path_edges))

    def get_context_from_neighborhood(self,topic_nodes,layer='lore',context_label='context'):

        # Extract relevant Lore / Plot nodes from the neighborhood of Topic nodes
        lore_info = {self.get_node_name(topic_node):[self.get_node_name(lore_node) for lore_node in self.collect_lore_nodes(topic_node)] for topic_node in topic_nodes}
        print("\nLore Info :",lore_info)           

        lore_nodes = []
        lore_edges = []
        for node in topic_nodes:
            lore_nodes += self.collect_lore_nodes(node) 

            node_id = self.get_node_id(node)
            lore_edges += [(head,tail) for head,tail in self.knowledge_graph.edges(node_id)]

        formatted_context = self.format_context(lore_info,"neighborhood",label=context_label)

        return formatted_context, list(set(lore_nodes)), list(set(lore_edges))

    def get_graph_rag_context(self,
                            user_query,
                            strategy='explore_neighborhood',
                            categories=None,
                            threshold=0.5,
                            k=10,
                            n_hops=1,
                            buffer_context=2,
                            add_code=False,
                            context_label='context'):

        # Convert user_query to a list of atomic sentences decomposed by sem.advanced_split_and_repair

        nodes_in_results = []
        for query in user_query:
            nodes_in_results += self.find_similar_nodes(query,threshold=threshold,k=k)
        nodes_in_results = list(set(nodes_in_results))

        if(strategy=='explore_neighborhood'):

            # Get the closest `topic` nodes from the `lore` / `plot` nodes
            topic_nodes = []
            for node in nodes_in_results:
                topic_nodes += self.get_related_topics(node)

            print(topic_nodes)

            topic_nodes = list(set(topic_nodes))
            print(topic_nodes)

            neighborhood = []
            for node in topic_nodes:
                neighborhood = self.explore_neighborhood(node,n_hops=n_hops,layer='topic')

            relevant_nodes = list(set(topic_nodes+neighborhood))

            context, lore_nodes, lore_edges = self.get_context_from_neighborhood(relevant_nodes,context_label=context_label)

            relevant_nodes += lore_nodes
            relevant_edges = lore_edges

        if(strategy=='find_path_between'):
            # Does it make sense to find a path between `topic` nodes?

            nodes_in_results = []
            for query in user_query:
                nodes_in_results += self.find_similar_nodes(query,threshold=threshold,k=k,filter='plot')

            print(f"NIR Before : {nodes_in_results}")
            nodes_in_results = list(set(nodes_in_results))
            print(f"NIR After : {nodes_in_results}")

            plot_paths = self.find_all_unique_paths(nodes_in_results,buffer_context=buffer_context)

            print("plot_paths",plot_paths)

            context, relevant_nodes, relevant_edges = self.get_context_from_paths(plot_paths,add_code=add_code,context_label=context_label)

        return context, relevant_nodes, relevant_edges

    def get_plot_to_reorder(self,nodes):

        info_dict = dict()
        node_dict = dict()
        for node in nodes:
            nodecode = generate_alphanumeric_code()
            info_dict[nodecode] = self.knowledge_graph.nodes[node]['text']
            node_dict[nodecode] = node
       
        context_to_reorder = self.format_for_reordering(info_dict)

        return context_to_reorder,node_dict

    def embed_text(self,text):
        return self.text_embedding_model.encode(text, normalize_embeddings=True)

    def get_cosine_similarity(self,source_text,target_text=None):
        # Both source text and target text must be lists
        # source_text size = S
        # target_text size = T

        source_embeddings = torch.from_numpy(self.text_embedding_model.encode(source_text, normalize_embeddings=True))
        if(target_text):
            target_embeddings = torch.from_numpy(self.text_embedding_model.encode(target_text, normalize_embeddings=True))
            cosine_sim = torch.matmul(source_embeddings,target_embeddings.T)
        else:
            cosine_sim = torch.matmul(source_embeddings,source_embeddings.T)

        # Cosine sim shape = S x T
        return cosine_sim

    def compare_against_existing_nodes(self,source_text: list, threshold=0.8)-> dict:
        # source_text is a list of strings containing all the text to be compared.

        if(len(self.dataset)>0):
            source_embeddings = torch.from_numpy(self.text_embedding_model.encode(source_text, normalize_embeddings=True))
            existing_embeddings = torch.stack([torch.tensor(row['embeddings']) for row in self.dataset])
            
            cosine_sim = torch.matmul(source_embeddings,existing_embeddings.T)

            comparison_dict = dict()
            for source_idx in range(len(cosine_sim)):
                comparison_dict[source_idx] = []
                for existing_idx in range(len(cosine_sim[0])):
                    if(cosine_sim[source_idx][existing_idx]>threshold):
                        comparison_dict[source_idx].append(self.dataset['node_id'][existing_idx])
        else:
            comparison_dict = dict()

        return comparison_dict

    def format_for_reordering(self,info_dict):
        context_text = f"\n<events>\n{"\n".join([f"[{key}] : {text}" for key,text in info_dict.items()])}\n</events>\n\n"

        return context_text
            
    def format_context(self,info_dict,mode,add_code=False,label='context'):

        context_text = ""
        if(mode=="neighborhood"):
            context_text = f"<{label}>\n"
            for topic,info in info_dict.items():
                context_text += f"\n<topic>\n{topic}\n</topic>"
                context_text += f"\n<information>\n{"\n".join([f"* {text}" for text in info])}\n</information>\n\n"
            context_text += f"</{label}>"
        if(mode=="path"):
            context_text = f"<{label}>\n"
            for path_idx,path in info_dict.items():
                context_text += f"\n[Phase No. {int(path_idx)+1}]"
                if(add_code):
                    context_text += f"\n<events>\n{"\n".join([f"{text}" for idx,text in enumerate(path)])}\n</events>\n\n"
                else:
                    context_text += f"\n<events>\n{"\n".join([f"{str(idx+1)}. {text}" for idx,text in enumerate(path)])}\n</events>\n\n"
            context_text += f"</{label}>"

            
        return context_text

    def create_network_figure(self, target_nodes, target_edges=None, add_edges=True, focus_on = None):
        # target_nodes : list(node_ids)
        # target_edges : list((head,tail,key))

        # Create a subgraph with the target nodes and target edges
        marker_size = 20

        if(focus_on==None):
            focus_on = dict()
            focus_on['nodes'] = []
            focus_on['edges'] = []

        focus_on['nodes'] += target_nodes
        all_neighbors = []
        for node in target_nodes:
            all_neighbors += self.knowledge_graph.predecessors(node)
            all_neighbors += self.knowledge_graph.successors(node)
            
        all_neighbors = list(set(all_neighbors))
        #all_neighbors = []

        G = self.knowledge_graph.subgraph(target_nodes+all_neighbors).copy()
        # 2. Extract the edge-induced subgraph

        
        pos = nx.spring_layout(G)
        
        # 2. Extract unique edges and aggregate data
        focus_edge_x, focus_edge_y = [], []
        edge_x, edge_y = [], []
        mid_x, mid_y, edge_text = [], [], []
        endpoints = []
        seen_edges = set()

        other_edge_x, other_edge_y = [], []
        
        # Initialize list to store arrow annotations
        annotations = []

        for u, v, data in G.edges(data=True):
            # Sort tuple for undirected graphs to avoid handling (A,B) and (B,A) separately
            edge_key = tuple(sorted([u, v]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)

                endpoints.append((u,v))

                x0, y0 = pos[u]
                x1, y1 = pos[v]

                u_pos = (x0*(self.fig_width),y0*(self.fig_height))
                v_pos = (x1*(self.fig_width),y1*(self.fig_height))

                standoff_value = math.dist(u_pos,v_pos)


                if(u in target_nodes or v in target_nodes):

                    #label = self.knowledge_graph.edges[u,v]['text']
                    #print(u,v,label)
                    arrow_color = "#EEE"
                    
                    if(not(u==v)):
                        edges = G.adj[u][v]
                        print('Edges : ',edges)
                        pair_text = [f"<i>{edges[key]['type']}</i> : {edges[key]['text']}" for key in edges.keys()]
                    else:
                        pair_text = []

                    print(pair_text)


                    focus_edge_x.extend([x0, x1, None])
                    focus_edge_y.extend([y0, y1, None])
                    # Add arrowhead annotation for primary target edges
                    annotations.append(
                        dict(
                            x=x1, y=y1,      # Arrow head target
                            ax=x0, ay=y0,    # Arrow tail start
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,     # Style of arrowhead (1 to 8)
                            arrowsize=1.0,   # Size multiplier
                            arrowwidth=2.2,  # Line width matching edge trace
                            arrowcolor='red',
                            standoff = standoff_value
                        )
                        )
                    
                    # Midpoint coordinates for edge hover triggering
                    if(not(u==v)):
                        mid_x.append((x0 + x1) / 2)
                        mid_y.append((y0 + y1) / 2)
                        edge_text.append("<br>".join(textwrap.wrap("<br>".join([text for text in pair_text]), width=36)))
                else:
                    other_edge_x.extend([x0, x1, None])
                    other_edge_y.extend([y0, y1, None])

                    # Add arrowhead annotation for secondary/other edges
                    annotations.append(
                        dict(
                            x=x1, y=y1,
                            ax=x0, ay=y0,
                            xref="x", yref="y",
                            axref="x", ayref="y",
                            showarrow=True,
                            arrowhead=2,
                            arrowsize=0.8,
                            arrowwidth=2.0,
                            arrowcolor="#EEE", # Match your secondary edge color
                            standoff = standoff_value
                        )
                    )


        focus_edge_trace = go.Scatter(
          x=focus_edge_x,
          y=focus_edge_y,
          line=dict(width=3.0, color="#d50e11"),
          hoverinfo="none",
          mode="lines"
        )

        edge_trace = go.Scatter(
          x=edge_x,
          y=edge_y,
          line=dict(width=2.0, color="#EEE"),
          hoverinfo="none",
          mode="lines"
        )

        other_edge_trace = go.Scatter(
          x=other_edge_x,
          y=other_edge_y,
          line=dict(width=2.0, color="#EEE"),
          hoverinfo="none",
          mode="lines"
        )

        # Transparent marker trace placed at midpoints to capture edge hover events
        edge_hover_trace = go.Scatter(
          x=mid_x,
          y=mid_y,
          mode="markers",
          hoverinfo="text",
          text=edge_text,
          marker=dict(opacity=0, size=20),  # Invisible, but sized to easily catch the mouse
        )

        # 4. Extract node coordinates and attributes for Plotly markers
        node_x = []
        node_y = []
        node_text = []
        node_ids = []
        node_labels = []
        node_colors = []
        border_widths = []
        border_colors = []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            wrapped_desc = ""
            node_text.append("<b>"+"<br>".join(textwrap.wrap(self.get_node_name(node[0]), width=20))+"</b><br>"+"<br>".join(textwrap.wrap(self.knowledge_graph.nodes[node[0]]['type'], width=20))+"<br>"+"Tags : <i>"+"<br>".join(textwrap.wrap(", ".join(self.knowledge_graph.nodes[node[0]]['tags']), width=20))+"</i>")
            node_ids.append(node[0])

            if(len(focus_on['nodes'])>0):

                if(node[0] in focus_on['nodes']):
                    border_widths.append(5)
                    border_colors.append('red')
                else:
                    border_widths.append(0)
                    border_colors.append("#444444")
            else:
                border_widths.append(0)
                border_colors.append("#444444")
                # Color code based on layer

            node_colors.append("#EEE")
            node_labels.append("")
            


        node_trace = go.Scatter(
          x=node_x,
          y=node_y,
          mode="markers+text",
          textposition="bottom center",
          hoverinfo="text",
          hovertext=node_text,
          text=node_labels,
          textfont=dict(
                color="#EEE",  
                size=marker_size,  
                family=('system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif')
            ),
          marker=dict(
                showscale=False,
                color="#EEE",
                line = dict(
                        width = border_widths,
                        color = border_colors
                    ),
                size=30)
        )

        node_trace.marker.color = node_colors
        node_trace.text = node_labels

        # 2. Extract coordinates for just these nodes
        subset_x = [pos[node][0] for node in target_nodes+all_neighbors if node in pos]
        subset_y = [pos[node][1] for node in target_nodes+all_neighbors if node in pos]

        # 3. Calculate bounding box ranges with a 20% padding margin
        if subset_x and subset_y:
          x_min, x_max = min(subset_x), max(subset_x)
          y_min, y_max = min(subset_y), max(subset_y)

          x_pad = (x_max - x_min) * 0.2 if x_max != x_min else 1.0
          y_pad = (y_max - y_min) * 0.2 if y_max != y_min else 1.0

          x_range = [x_min - x_pad, x_max + x_pad]
          y_range = [y_min - y_pad, y_max + y_pad]
        else:
          x_range, y_range = None, None

        # 5. Build the final Plotly figure with annotations included
        fig = go.Figure(
          data=[focus_edge_trace, edge_trace, other_edge_trace],
          layout=go.Layout(
              showlegend=False,
              hovermode="closest",
              plot_bgcolor="rgba(0,0,0,0)",  
              paper_bgcolor="rgba(0,0,0,0)",  
              annotations=annotations, # <-- Added arrow annotations here
              hoverlabel=dict(
                    bgcolor="#ffffff",  
                    font_color="#000000",  
                    font_size=12,  
                    font_family=(
                        'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",'
                        ' Roboto, Helvetica, Arial, sans-serif'
                    ), 
                    bordercolor="#ffffff",  
                ),
              dragmode="pan",
              margin=dict(b=20, l=5, r=5, t=40),
              xaxis=dict(
                range=x_range, showgrid=False, zeroline=False, showticklabels=False
              ),
              yaxis=dict(
                range=y_range, showgrid=False, zeroline=False, showticklabels=False
              ),
          ),
        )

        fig.update_annotations(startstandoff=marker_size)
        fig.add_trace(edge_hover_trace)
        fig.add_trace(node_trace)
        return fig








