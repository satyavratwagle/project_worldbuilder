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
from datetime import datetime, date
import matplotlib.pyplot as pltk
import itertools
import difflib
import time
import random
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

# Patch to prevent AttributeError with older packages on Transformers 5.x
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr
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

class DatastoreUtilities():

    def __init__(self,config,name="worldbuilding",load_most_recent=True):

        self.config = config

        self.kg_to_ds_path = config['kg_to_ds_map']
        self.ds_name = name
        self.knowledge_graph_prefix = f"{self.config['graph_dir']}/{self.ds_name}_"
        self.datastore_prefix = f"{self.config['data_dir']}/FAISS_store/{self.ds_name}_"

        if(not(os.path.exists(self.kg_to_ds_path))):
            # kg_to_ds_path should indicate the knowledge graph version contained in the current FAISS dataset and the ones not tracked
            kg2ds = dict()
            kg2ds['datastore_version'] = None
            kg2ds['datastore_last_saved'] = 0
            kg2ds['unsaved_versions'] = []
            self.kg2ds_map = kg2ds
        else:
            with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
                self.kg2ds_map = json.load(f)


        # First check for the knowledge graph

        if(len(self.kg2ds_map['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")):
            # load the latest knowledge graph
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")
            self.edge_key = (max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])+1) if len(self.knowledge_graph.edges(keys=True))>0 else 1
            self.kg_loaded_version = self.kg2ds_map['unsaved_versions'][-1]
        else:
            kg_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.knowledge_graph = nx.MultiGraph()
            nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_prefix}{kg_suffix}.gml")
            self.kg2ds_map['unsaved_versions'].append(kg_suffix)
            self.edge_key = 1
            self.kg_loaded_version = kg_suffix

        print(f"Loaded Graph Version : {self.kg_loaded_version} : {self.knowledge_graph}")
        
        if(self.kg2ds_map['datastore_version'] and not(os.path.exists(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}"))):
            self.kg2ds_map['datastore_version'] = None

        if(self.kg2ds_map['datastore_version']):
            self.dataset = datasets.load_from_disk(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}")
            self.dataset.load_faiss_index("embeddings", f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}.faiss")
        else:
            # No datastore was found
            self.dataset = None

        # Update the kg_to_ds_map
        with open(config['kg_to_ds_map'], "w") as f:
            json.dump(self.kg2ds_map, f, indent=4)

        alias_pattern = rf"\b({'|'.join(re.escape(alias[1]['name']) for alias in self.knowledge_graph.nodes(data=True))})\b"
        self.node_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

        # CLEAN UP OLD KNOWLEDGE GRAPHS
            
    # FAISS DATASET FUNCTIONS

    def create_dataset_from_nodes(self,nodes,get_index=False):
        # Create a dataset from a set of nodes 
        # Use get_index if generating dataset for the first time.

        documents = []
        for node in nodes:
            update_time = int(time.time())
            for key,statement in enumerate(self.get_node_summary(node[0])):
                documents.append({
                                    'topic':node[1]['name'],
                                    'type':node[1]['type'],
                                    'node_id':node[0],
                                    'id' : f"{node[0]}-{key}",
                                    'text':statement
                                })

        graph_dataset = Dataset.from_list(documents)

        return graph_dataset

    def filter_dataset(self):

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}

        self.save_graph()
        nodes_to_add = self.get_nodes_to_add_to_faiss()

        if(self.dataset):
            nodes_to_remove = [row['node_id'] for row in self.dataset if not(self.knowledge_graph.has_node(row['node_id']))]
        else:
            nodes_to_remove = []

        if(len(nodes_to_add)>0 or len(nodes_to_remove)>0):

            if(self.dataset):

                nodes_to_remove = list(set(nodes_to_remove))

                # Create a new dataset based on the new dataset
                new_dataset = self.create_dataset_from_nodes(nodes_to_add)
                print("NEW DATASET SIZE : ",len(new_dataset))
                for row in new_dataset:
                    print(f"{row['id']} - {row['node_id']} : {row['text']}")
                print()

                # Add embeddings to new dataset
                new_dataset = new_dataset.map(embed_text, batched=True, batch_size=8)

                print("CURRENT DATASET SIZE : ",len(self.dataset))
                # Filter out data from existing dataset coming from the same nodes
                if "embeddings" in self.dataset.list_indexes():
                    self.dataset.drop_index(index_name="embeddings")
                filtered_dataset = self.dataset
                for node in nodes_to_add:
                    filtered_dataset = filtered_dataset.filter(lambda example: example["node_id"] != node[0])

                # Check if all nodes in filtered_dataset are in the current knowledge graph
                for node in nodes_to_remove:
                    filtered_dataset = filtered_dataset.filter(lambda example: example["node_id"] != node)


                print("FILTERED DATASET SIZE : ",len(filtered_dataset))

                for row in filtered_dataset:
                    print(f"{row['id']} - {row['node_id']} : {row['text']}")
                print()

                updated_dataset = concatenate_datasets([filtered_dataset,new_dataset])
                print("UPDATED DATASET SIZE : ",len(updated_dataset))

            else:
                updated_dataset = self.create_dataset_from_nodes(nodes_to_add)
                updated_dataset = updated_dataset.map(embed_text, batched=True, batch_size=8)

            embedding_dim = self.text_embedding_model.get_embedding_dimension()
            cosine_index = faiss.IndexFlatIP(embedding_dim)

            return updated_dataset, cosine_index

        else:

            return None, None

    def overwrite_faiss_dataset(self,new_dataset,cosine_index,temp_name='_temp'):
        
        if(new_dataset):
            temp_dataset_path = f"{self.config['data_dir']}/FAISS_store/{temp_name}"
            temp_faiss_path = f"{self.config['data_dir']}/FAISS_store/{temp_name}.faiss"

            new_dataset.info.description = str(int())

            # Save to temporary location
            new_dataset.save_to_disk(temp_dataset_path)
            new_dataset.add_faiss_index(column="embeddings", custom_index=cosine_index, index_name="faiss_cosine")

            if os.path.exists(temp_faiss_path):
                os.remove(temp_faiss_path)
            new_dataset.save_faiss_index("faiss_cosine", temp_faiss_path)

            # Delete old dataset if exists
            for folder in os.listdir(f"{self.config['data_dir']}/FAISS_store/"):
                if(os.path.isdir(f"{self.config['data_dir']}/FAISS_store/{folder}") and not(folder==temp_name)):
                    shutil.rmtree(f"{self.config['data_dir']}/FAISS_store/{folder}")
                    os.remove(f"{self.config['data_dir']}/FAISS_store/{folder}.faiss")

            # Rename temporary location
            os.rename(temp_dataset_path,f"{self.datastore_prefix}{self.kg_loaded_version}")
            os.rename(temp_faiss_path,f"{self.datastore_prefix}{self.kg_loaded_version}.faiss")

            self.kg2ds_map['datastore_version'] = self.kg_loaded_version
            self.kg2ds_map['unsaved_versions'] = [self.kg_loaded_version]

            # Keep only the latest graph
            for graph in os.listdir(f"{self.config['graph_dir']}"):
                if(not(graph==f"{self.ds_name}_{self.kg_loaded_version}.gml")):
                    os.remove(f"{self.config['graph_dir']}/{graph}")

            self.kg2ds_map['datastore_last_saved'] = int(time.time())

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

        if(self.kg2ds_map['datastore_version'] and os.path.exists(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}")):
            self.dataset = datasets.load_from_disk(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}")
            self.dataset.load_faiss_index("embeddings", f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}.faiss")
        else:
            self.dataset = None
        print(self.dataset)
        for row in self.dataset:
            print(f"{row['id']} - {row['node_id']} : {row['text']}")

    '''
    def calculate_faiss_index(self,dataset):

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}
                
        embedding_dim = self.text_embedding_model.get_embedding_dimension()
        cosine_index = faiss.IndexFlatIP(embedding_dim)
        dataset = dataset.map(embed_text, batched=True, batch_size=8)

        return dataset, cosine_index

    def update_faiss_dataset(self):

        # Need a warning here if the current graph has not been saved.
        self.datastore_save_flag = self.save_graph()
        if(self.datastore_save_flag):
            self.reload_graph(most_recent=True)
            self.nodes_to_add = self.get_nodes_to_add_to_faiss()
            new_dataset,_ = self.create_dataset_from_nodes(self.nodes_to_add)

            print([doc['id'] for doc in new_dataset])
            
            if(os.path.exists(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}")):
                # Filter out existing data from old dataset
                old_dataset = datasets.load_from_disk(f"{self.datastore_prefix}{self.kg2ds_map['datastore_version']}")

                # Keep the new updated dataset values
                filtered_dataset = old_dataset
                filtered = False
                for doc in new_dataset:
                    filtered_dataset = filtered_dataset.filter(lambda example: example["id"] != doc['id'])
                    filtered = True

                # Remove any deleted files
                old_topics = list(set([doc['id'] for doc in old_dataset]))
                excluded_topics = []
                for topic in old_topics:
                    node_id,key = topic.split('-')
                    if (self.knowledge_graph.has_node(node_id)):
                        if(not(key in self.knowledge_graph.nodes[node_id]['wiki'].keys())):
                            excluded_topics.append(topic)
                    else:
                        excluded_topics.append(topic)
                    
                for excluded_topic in excluded_topics:
                    filtered_dataset = filtered_dataset.filter(lambda example: example["id"] != excluded_topic)
                    
                print("OLD DATASET SIZE : ",len(old_dataset))
                print("FILTERED DATASET SIZE : ",len(filtered_dataset))
                print("NEW DATASET SIZE : ",len(new_dataset))

                # Append datasets.
                updated_dataset = concatenate_datasets([filtered_dataset,new_dataset])
                print("UPDATED DATASET SIZE : ",len(updated_dataset))
            else:
                updated_dataset = new_dataset

            updated_dataset,updated_index = self.calculate_faiss_index(updated_dataset)

            return updated_dataset,updated_index
        else:
            print("No new data found in knowledge graph!")
            return None, None
    '''

    # EDGE FUNCTIONS

    def add_edge(self,head,tail,text):
        # Add an edge to the graph
        tags_dict = dict()
        tags_dict[slugify_key(head)] = []
        tags_dict[slugify_key(tail)] = []
        time_now = int(time.time())
        self.knowledge_graph.add_edge(slugify_key(head), slugify_key(tail),desc=text,parsed=False,key=str(self.edge_key),tags=tags_dict, updated=time_now, endpoints=[head,tail])
        self.edge_key += 1

        return str(self.edge_key-1)

    def remove_edge(self,head,tail,key):
        if(self.knowledge_graph.has_edge(head,tail,key=key)):
            self.knowledge_graph.remove_edge(head,tail,key=key)

    def get_edges_from(self,head):
        return [edge for edge in self.knowledge_graph.edges(head,keys=True,data=True)]

    def get_edge(self,head,tail):
        # Get all edges between a head and tail node
        return [(head,tail,k,v['desc'],v['endpoints']) for k,v in self.knowledge_graph.adj[head][tail].items()]

    def parse_edge(self,head,tail,key):
        self.knowledge_graph.edges[head,tail,key]['parsed'] = True

    def get_relevant_edges(self,text,threshold=0.8):
        # All relevant edges are structured as list((node,tail,info))

        graph_edges = list(self.knowledge_graph.edges(keys=True,data=True))
        text_embedding = torch.from_numpy(self.text_embedding_model.encode([text], normalize_embeddings=True))
        embeddings = torch.from_numpy(self.text_embedding_model.encode([e[3]['desc'] for e in graph_edges], normalize_embeddings=True))
        cosine_sim = torch.matmul(embeddings,text_embedding.T)

        similar_idxs = torch.nonzero(torch.flatten(cosine_sim)>threshold,as_tuple=True)[0]

        all_info = dict()
        if(len(similar_idxs)>0):
            all_info['node_name'] = f"Edges similar to '{text}'"
            all_info['summary'] = None
            all_info['edge_info'] = [(graph_edges[idx][0],graph_edges[idx][1],graph_edges[idx][2],graph_edges[idx][3]['desc']) for idx in similar_idxs]
        else:
            all_info = None

        return all_info

    def set_edge_tags(self,edge,topic,tag):
        head,tail,key,metadata = edge
        self.knowledge_graph.edges[head,tail,key]['tags'][topic] = [tag]
        self.knowledge_graph.edges[head,tail,key]['updated'] = int(time.time())

    def get_num_edges(self):
        # Return the number of edges in the graph
        return max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])

    def resolve_edges(self,threshold=0.9):
        # Remove redundant edges and add new connections based on edge data if any
        
        # Remove Redandant Edges
        print(self.knowledge_graph)
        edges_to_resolve = []
        idxs_clustered = []
        for node,_ in self.knowledge_graph.nodes(data=True):

            # Calculate cosine similarity between outgoing edges for each node.
            edge_descriptions = [(e[0],e[1],e[2],e[3]['desc']) for e in self.knowledge_graph.edges(node,keys=True,data=True)]
            embeddings = torch.from_numpy(self.text_embedding_model.encode([e[3] for e in edge_descriptions], normalize_embeddings=True))
            cosine_sim = torch.matmul(embeddings,embeddings.T)

            for idx in range(len(edge_descriptions)):
                if(not(idx in idxs_clustered)):
                    similar_idxs = torch.nonzero(cosine_sim[idx]>threshold,as_tuple=True)[0]
                    edges_to_resolve.append([edge_descriptions[sim_idx] for sim_idx in similar_idxs])
                    idxs_clustered += [sim_idx for sim_idx in similar_idxs if not(sim_idx in idxs_clustered)]

        #self.knowledge_graph.remove_edges_from(edges_to_remove)
        #print(self.knowledge_graph)

        return edges_to_resolve

    def find_new_edges(self):

        # Check for any new connections

        # Get aliases of all nodes
        node_aliases = self.get_all_nodes()

        alias_pattern = rf"\b({'|'.join(re.escape(alias['name']) for alias in node_aliases)})\b"
        alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

        added_edges = []
        edges_log = []
        for edge in self.knowledge_graph.edges(keys=True,data=True):
            desc = edge[3]['desc']
            src_key = edge[2]

            relevant_nodes = [slugify_key(name) for name in list(set(alias_finder.findall(desc))) if len(name)>0]

            if(len(relevant_nodes)>1):
                pairs = [(relevant_nodes[jdx],relevant_nodes[idx]) for idx in range(1,len(relevant_nodes)) for jdx in range(idx)]
                
                for head,tail in pairs:
                    already_added = False
                    if(tail in self.knowledge_graph.neighbors(head)):
                        edge_keys = [edge[2] for edge in self.knowledge_graph.edges([head,tail],keys=True,data=True)]
                        if(src_key in edge_keys):
                            already_added = True

                    if(not(already_added)):
                        added_edges.append((head,tail,desc))
                        edges_log.append((self.knowledge_graph.nodes[head]['name'],self.knowledge_graph.nodes[tail]['name'],desc))
                        print(edge)
                        print("HEre :",src_key)
                        print(f"Added edge '{desc}' between {head} and {tail}")

        for head,tail,desc in added_edges:
            self.add_edge(head,tail,desc)

        print(self.knowledge_graph)
        return edges_log

    # NODE FUNCTIONS

    def add_node(self,node,type,summary=[],definition=""):

        # Add a node to the graph
        if(not(self.knowledge_graph.has_node(slugify_key(node)))):

            self.knowledge_graph.add_node(slugify_key(node),
                                            updated=int(time.time()),
                                            name=node,
                                            type=type,
                                            summary=summary,
                                            definition="")

        if(len(summary)>0):
            assert type(summary)==list
            self.knowledge_graph.nodes[node]['summary'] = summary
                
    def get_node_name(self,node_id):
        print(node_id)
        print(self.knowledge_graph.nodes[node_id])
        return self.knowledge_graph.nodes[node_id]['name']

    def remove_node(self,node):
        # Cleanly remove a node from the knowledge graph

        print(self.knowledge_graph)
        if(self.knowledge_graph.has_node(node)):
            current_edges = list(self.knowledge_graph.edges([node],keys=True,data=True))
            for idx in range(len(current_edges)):
                #print(current_edges[idx])
                if(not(current_edges[idx][1]==node)):
                    # Make the current edge self loop onto tail
                    edge = list(current_edges[idx])
                    edge[0] = edge[1]
                    self.knowledge_graph.update(edges=[tuple(edge)])
                else:
                    # Otherwise delete the edge altogether
                    self.knowledge_graph.remove_edge(current_edges[idx][0],current_edges[idx][1],key=current_edges[idx][2])

            self.knowledge_graph.remove_node(node)

        print(self.knowledge_graph)

    def get_all_nodes(self):
        # Return a list of all node names
        return [{"name":node[1]['name'],"id":node[0]} for node in self.knowledge_graph.nodes(data=True)]

    def get_all_node_data(self):
        nodes = self.get_all_nodes()
        node_descriptions = dict()
        for node in nodes:
            node_descriptions[node['id']] = list(self.knowledge_graph.edges(node['id'],keys=True,data=True))
        return node_descriptions

    def is_node_updated(self,node):
        edge_data = self.knowledge_graph.edges(node,data=True)
        return not(any([self.knowledge_graph.nodes[node]['updated']<e[-1]['updated'] for e in edge_data]))

    def get_nodes_to_add_to_faiss(self):
        # Return a list of nodes that have been updated, but not added to the FAISS dataset
        # Decompose a node into wiki keys and check if the ids are in the dataset.

        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        if(not(self.dataset)):
            return [node for node in self.knowledge_graph.nodes(data=True)]
        else:
            nodes_in_dataset = set(self.dataset['id'])
            print([node for node in self.knowledge_graph.nodes(data=True) if node[1]['updated']>self.kg2ds_map['datastore_last_saved']])
            return [node for node in self.knowledge_graph.nodes(data=True) if node[1]['updated']>self.kg2ds_map['datastore_last_saved']]

    def get_random_node(self):
        all_nodes = self.get_all_nodes()
        return random.choice(all_nodes)

    def get_neighbors(self,node):
        return [n for n in self.knowledge_graph.neighbors(node)]

    def reset_nodes(self):
        # Reset if the nodes need to be saved to the dataset all over again.

        for node_data in self.get_all_nodes():
            node_id = node_data['id']
            self.knowledge_graph.nodes[node_id]['updated']  = int(time.time())

    def check_if_node_exists(self,node):
        return self.knowledge_graph.has_node(slugify_key(node))

    def has_new_edges(self,node):

        needs_update = any([edge[3]['updated']>self.knowledge_graph.nodes[node]['updated'] for edge in self.get_edges_from(node)])

        return needs_update

    def cluster_edge_info(self,node):

        node = slugify_key(node)

        all_edge_desc = [edge[3]['desc'] for edge in self.get_edges_from(node)]

        edge_embeddings = self.text_embedding_model.encode(all_edge_desc, normalize_embeddings=True)

        if(len(edge_embeddings)>1):
            best_k = 2
            best_score = -1
            for num_clusters in range(2,min((6,len(edge_embeddings)))):
                skm = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, random_state=42)
                skm.fit(edge_embeddings)

                labels = skm.labels_
                score = silhouette_score(edge_embeddings, labels, metric='cosine')

                if score > best_score:
                    best_score = score
                    best_k = num_clusters

            skm = KMeans(n_clusters=best_k, init='k-means++', n_init=10, random_state=42)
            skm.fit(edge_embeddings)
            labels = skm.labels_

            edge_clusters = [[] for i in range(best_k)]

            for idx,l in enumerate(labels):
                edge_clusters[l].append(all_edge_desc[idx])
        else:
            edge_clusters = [[all_edge_desc[0]]]

        for idx in range(len(edge_clusters)):
            edge_clusters[idx] = "\n".join(edge_clusters[idx])

        return edge_clusters

    def get_node_info(self,node):

        # All info is structured as {node_name, node_summary, list((node,tail,info))}

        node = slugify_key(node)
        if(self.knowledge_graph.has_node(node)):
            neighbors = list(self.knowledge_graph.adj[node])

            all_info = dict()

            all_info['id'] = node
            all_info['node_name'] = self.knowledge_graph.nodes[node]['name']
            all_info['node_type'] = self.knowledge_graph.nodes[node]['type']
            all_info['summary'] = self.knowledge_graph.nodes[node]['summary']
            all_info['definition'] = self.knowledge_graph.nodes[node]['definition']
            all_info['edge_info'] = []
            for neighbor in neighbors:
                all_info['edge_info'] += self.get_edge(node, neighbor)

            return all_info
        else:
            return None

    def get_node_id(self,node):
        return slugify_key(node)

    def set_node_definition(self,node, definition):
        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['definition'] = definition
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            print(f"Set Definition of {node} : ",self.knowledge_graph.nodes[node]['definition'])
            return True

        return False

    def get_node_definition(self,node):
        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            return self.knowledge_graph.nodes[node]['definition']

    def reset_node_summary(self,node):

        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['summary'] = []
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            return True

        return False

    def add_to_node_summary(self,node,summary):

        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['summary'].append(summary)
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            print(f"Set Summary of {node} : ",self.knowledge_graph.nodes[node]['summary'])
            return True

        return False
    
    def get_node_summary(self,node):
        return self.knowledge_graph.nodes[slugify_key(node)]['summary'] if self.knowledge_graph.has_node(slugify_key(node)) else []
    
    # GRAPH FUNCTIONS

    def get_relevant_nodes(self,query,k=10, threshold=0.4):

        # Check literal mentions of any node names.
        relevant_nodes = [slugify_key(name) for name in list(set(self.node_finder.findall(query))) if len(name)>0]

        query_vector = self.embed_text(query)
        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=k)

        for i in range(len(scores)):
            if(scores[i] >= threshold and len(examples['text'][i].split(':')[-1].strip())>0):
                #print(f"{scores[i]} : {examples['node_id'][i]} - {examples['text'][i]}")
                relevant_nodes.append(examples['node_id'][i])

        relevant_nodes = list(set(relevant_nodes))
        print("Relevant Nodes : ",relevant_nodes)

        return relevant_nodes

    def extract_relevant_edge_info(self,query,node,threshold=0.7):

        node = slugify_key(node)
        print(node)

        documents = []
        
        for edge in self.knowledge_graph.edges(node,keys=True,data=True):
            documents.append({"text":edge[3]['desc']})

        edge_dataset = Dataset.from_list(documents)
        edge_dataset, cosine_index = self.calculate_faiss_index(edge_dataset)
        edge_dataset.add_faiss_index(
                column="embeddings", 
                custom_index=cosine_index
            )

        query_vector = self.embed_text(query)
        scores, examples = edge_dataset.get_nearest_examples("embeddings", query_vector, k=20)

        relevant_edge_info = []
        for i in range(len(scores)):
            if(scores[i] >= threshold and len(examples['text'][i].split(':')[-1].strip())>0):
                relevant_edge_info.append(examples['text'][i])

        return relevant_edge_info

    def find_all_unique_paths(self,nodes):
        all_paths = []
        for idx in range(len(nodes)):
            for jdx in range(len(nodes)):
                if(not(idx==jdx)):
                    path = self.find_path(nodes[idx],nodes[jdx])
                    if(path):
                        all_paths.append(path)

        all_paths = sorted(all_paths,key=len,reverse=True)

        filtered_paths = []
        for path in all_paths:
            if(not(any([self.check_subpath(path,c_path) for c_path in filtered_paths]))):
                filtered_paths.append(path)
        filtered_paths = sorted(filtered_paths,key=len,reverse=True)

        return filtered_paths

    def find_all_unique_paths_from(self,head,nodes):

        all_paths = []
        for idx in range(len(nodes)):
            if(not(nodes[idx]==head)):
                path = self.find_path(head,nodes[idx])
                if(path):
                    all_paths.append(path)

        all_paths = sorted(all_paths,key=len,reverse=True)

        filtered_paths = []
        for path in all_paths:
            if(not(any([self.check_subpath(path,c_path) for c_path in filtered_paths]))):
                filtered_paths.append(path)
        filtered_paths = sorted(filtered_paths,key=len,reverse=True)

        return filtered_paths

    def extract_path_info(self,paths):

        added_nodes = []
        added_relations = []
        paths_info = []

        for path in paths:
            for idx in range(len(path)-1):
                # Extract head info

                if(idx==0):
                    # The first node is crucial, so more information is given about it.
                    paths_info.append(('node',path[idx],'\n'.join(self.get_node_summary(path[idx]))))
                    added_nodes.append(path[idx])
                else:
                    # Add definitions for the rest of the node
                    if(not(path[idx] in added_nodes)):
                        paths_info.append(('node',path[idx],self.get_node_definition(path[idx])))
                        added_nodes.append(path[idx])

                # Extract relation info
                if(not(f"{path[idx]}_and_{path[idx+1]}" in added_relations) and not(f"{path[idx+1]}_and_{path[idx]}" in added_relations)):
                    edge_info = "\n".join([edge[3] for edge in self.get_edge(path[idx],path[idx+1])])
                    paths_info.append(('edge',f"{path[idx]}_and_{path[idx+1]}",edge_info))
                    added_relations.append(f"{path[idx]}_and_{path[idx+1]}")

                # Extract tail info
                if(idx+1==len(path)):
                    # The final node is crucial, so more information is given about it.
                    paths_info.append(('node',path[idx+1],'\n'.join(self.get_node_summary(path[idx+1]))))
                    added_nodes.append(path[idx+1])
                else:
                    if(not(path[idx+1] in added_nodes)):
                        paths_info.append(('node',path[idx+1],self.get_node_definition(path[idx+1])))
                        added_nodes.append(path[idx+1])

        return paths_info

    def graph_multihop(self,node,n_hops=1,neighborhood=[]):

        if(len(neighborhood)==0):
            neighborhood = [node]

        if(n_hops>0):
            node_neighbors = self.knowledge_graph.neighbors(node)
            for neighbor in node_neighbors:
                if(not(neighbor in neighborhood)):
                    neighborhood.append(neighbor)
                    new_set = self.graph_multihop(neighbor,n_hops=(n_hops-1),neighborhood=neighborhood)
        return neighborhood

    def get_unparsed_edges(self,node):
        unparsed_edges = [edge for edge in self.knowledge_graph.edges(node,keys=True,data=True) if self.knowledge_graph.nodes[node]['updated']<edge[3]['updated']]
        return unparsed_edges

    def get_node_summaries(self,nodes):
        return [self.get_node_summary(node) for node in nodes]

    def check_nodes_for_replacement(self,named_entities,threshold=0.85):
        nodes = self.get_all_nodes()

        similar_pairs = []
        for key in named_entities:
            similarity = list(map(lambda x :difflib.SequenceMatcher(None, x,key).ratio(),[node['name'] for node in nodes]))
            similar_pairs += [(key,nodes[i]) for i, sim in enumerate(similarity) if (sim > threshold and not(key==nodes[i]))]
        
        return similar_pairs

    def find_path(self,source,target):

        source = slugify_key(source)
        target = slugify_key(target)

        path = None
        if(self.knowledge_graph.has_node(source) and self.knowledge_graph.has_node(target)):
            if nx.has_path(self.knowledge_graph, source, target):
                path = nx.shortest_path(self.knowledge_graph, source=source, target=target)

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

    # TODO ENDS

    def find_topics_in_text(self,text):

        topics = self.get_all_nodes()

        alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias in topics)})\b"
        alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

        topics_in_text = [name for name in list(set(alias_finder.findall(text))) if len(name)>0]

        return topics_in_text

    def save_graph(self,draw_figure=False):

        # Reload kg_to_ds_map
        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        if(len(self.kg2ds_map['unsaved_versions'])>0):
            # Check if graphs exist. If not, remove them from the map
            versions_to_remove = []
            for idx,kg_version in enumerate(self.kg2ds_map['unsaved_versions']):
                if(not(os.path.exists(f"{self.knowledge_graph_prefix}{kg_version}.gml"))):
                    versions_to_remove.append(kg_version)
            [self.kg2ds_map['unsaved_versions'].remove(version) for version in versions_to_remove]

            with open(self.config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

        if(len(self.kg2ds_map['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")):
            # load the latest knowledge graph
            most_recent_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")
        else:
            most_recent_graph = None

        if(True):
            print("Saving Graph!")
            # Clean up old graphs
            # 1 day = 86400 seconds

            self.kg_saved_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_prefix}{self.kg_saved_version}.gml")

            self.kg2ds_map['unsaved_versions'].append(self.kg_saved_version)

            with open(self.kg_to_ds_path, "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

            self.reload_graph()
            return True
        else:
            print('There are no changes to be saved!')
            return False

    def reload_graph(self,most_recent=True):

        # Reload kg_to_ds_map
        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        if(len(self.kg2ds_map['unsaved_versions'])>0):
            # Check if graphs exist. If not, remove them from the map
            versions_to_remove = []
            for idx,kg_version in enumerate(self.kg2ds_map['unsaved_versions']):
                if(not(os.path.exists(f"{self.knowledge_graph_prefix}{kg_version}.gml"))):
                    versions_to_remove.append(kg_version)
            [self.kg2ds_map['unsaved_versions'].remove(version) for version in versions_to_remove]

            with open(self.config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

        if(len(self.kg2ds_map['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")):
            # load the latest knowledge graph
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map['unsaved_versions'][-1]}.gml")
            self.edge_key = (max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])+1) if len(self.knowledge_graph.edges(keys=True))>0 else 1
            self.kg_loaded_version = self.kg2ds_map['unsaved_versions'][-1]
            print(f"Reloaded Graph Version : {self.kg_loaded_version} : {self.knowledge_graph}")
            return True
        else:
            return False

    def load_embedding_model(self):
        self.text_embedding_model = SentenceTransformer(self.config['text_embedding_model'])

    # RAG FUNCTIONS

    def get_context_from_neighborhood(self,neighborhood):

        neighborhood_context = [('node',node,self.get_node_definition(node)) for node in list(set(neighborhood))]
        formatted_context = self.format_path_context(neighborhood_context)

        return formatted_context


    def get_graph_rag_context(self,query,threshold=0.4,k=10):
        #def get_graph_rag_context(self,query,graph,documents_lookup,threshold=0.4,k=10,hops=1):

        # need self.knowledge_graph, self.documents_lookup

        print(query)
        nodes_in_results = self.get_relevant_nodes(query,threshold=threshold,k=k)

        retrieved_context = []
        edge_paths = []

        if(len(nodes_in_results)>1):
            
            edge_paths = self.find_all_unique_paths(nodes_in_results)
            retrieved_context = self.extract_path_info(edge_paths)

        elif(len(nodes_in_results)==0):
            retrieved_context = []
            edge_paths = [] 
        else:
            # If there is only one node in the results
            node = nodes_in_results[0]
            retrieved_context = [('node',node,'\n'.join(self.get_node_summary(node)))]
            edge_paths += [node]

        context_text = self.format_path_context(retrieved_context)
            
        return context_text, edge_paths

    def embed_text(self,text):
        return self.text_embedding_model.encode(text, normalize_embeddings=True)

    def get_cosine_similarity(self,source_text,target_text=None):
        # Both source text and target text must be lists

        source_embeddings = torch.from_numpy(self.text_embedding_model.encode(source_text, normalize_embeddings=True))
        if(target_text):
            target_embeddings = torch.from_numpy(self.text_embedding_model.encode(target_text, normalize_embeddings=True))
            cosine_sim = torch.matmul(source_embeddings,target_embeddings.T)
        else:
            cosine_sim = torch.matmul(source_embeddings,source_embeddings.T)

        return cosine_sim

    def format_path_context(self,paths):
        context_text = ""
        for p in paths:
            if(p[0]=='node'):
                context_text += f"<context_about_{p[1]}>\n{p[2]}\n</context_about_{p[1]}>\n\n"
            if(p[0]=='edge'):
                context_text += f"<relationship_between_{p[1]}>\n{p[2]}\n</relationship_between_{p[1]}>\n\n"

        return context_text
    # May not need functions beyond this.
    def get_most_relevant_file(self,query,threshold,k):

        query_vector = self.embed_text(query)
        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=k)
    
        if scores[0] >= threshold:
            return examples
        else:
            return []

    def filter_similar_text(self,text_list,cosine_similarity,threshold=0.8):
        # Text list should be what you want to filter the similar sentences from
        # Returns two lists, similar_text and unique_text

        similar_text = []
        unique_text = []

        omitted_idxs = []
        for idx in range(len(text_list)):
            if(not(idx in omitted_idxs)):
                similar_idxs = torch.nonzero(cosine_similarity[idx]>0.8,as_tuple=True)[0]
                for sim_idx in similar_idxs:
                    omitted_idxs.append(sim_idx)
                unique_text.append(text_list[idx].strip()+'.')

        similar_text = [text_list[idx] for idx in omitted_idxs]

        return unique_text,similar_text

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
        for d in self.dataset:
            documents_lookup[d['id']] = d['text']


        for d in self.dataset:
            graph.add_node(d['id'], type=d['type'])

        for d in self.dataset:
            if(len(d['text'].split(':')[-1].strip())>0):
                scores, neighbors = self.dataset.get_nearest_examples("embeddings", np.array(d['embeddings']), k=20)

                for i in range(len(scores)):
                    if (scores[i] >= 0.6 and not(d['id']==neighbors['id'][i])):
                        graph.add_edge(d['id'], neighbors['id'][i])

        return graph, documents_lookup

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

    def does_file_exist(self,name):
        entity = name.lower()
        filename = re.sub(r'[^a-zA-Z0-9]', '_', entity)
        exists = Path(f"{self.jsonstore_dir}/{filename}.json").exists()
        return exists
        
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
