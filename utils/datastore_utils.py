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
            kg2ds['unsaved_versions'] = []
            self.kg2ds_map = kg2ds
        else:
            with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
                self.kg2ds_map = json.load(f)


        # First check for the knowledge graph
        if(len(self.kg2ds_map['unsaved_versions'])>0):
            # Check if graphs exist. If not, remove them from the map
            versions_to_remove = []
            for idx,kg_version in enumerate(self.kg2ds_map['unsaved_versions']):
                if(not(os.path.exists(f"{self.knowledge_graph_prefix}{kg_version}.gml"))):
                    versions_to_remove.append(kg_version)
            [self.kg2ds_map['unsaved_versions'].remove(version) for version in versions_to_remove]

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

        # CLEAN UP OLD KNOWLEDGE GRAPHS
            
    # FAISS DATASET FUNCTIONS

    def create_dataset_from_nodes(self,nodes,get_index=False):
        # Create a dataset from a set of nodes 
        # Use get_index if generating dataset for the first time.

        with open('config.json', "r", encoding="utf-8") as f:
            self.config = json.load(f)

        documents = []
        for node in nodes:
            for key in node[1]['wiki'].keys():
                documents.append({'topic':node[1]['name'],
                                    'type':node[1]['type'],
                                    'tag':key,
                                    'node_id':node[0],
                                    'id' : f"{node[0]}-{key}",
                                    'text':node[1]['wiki'][key]})

        graph_dataset = Dataset.from_list(documents)

        cosine_index = None
        if(get_index):
            graph_dataset,cosine_index = self.calculate_faiss_index(graph_dataset)

        return graph_dataset,cosine_index

    def overwrite_faiss_dataset(self,new_dataset,cosine_index,temp_name='_temp'):

        if(self.datastore_save_flag):
            temp_dataset_path = f"{self.config['data_dir']}/FAISS_store/{temp_name}"
            temp_faiss_path = f"{self.config['data_dir']}/FAISS_store/{temp_name}.faiss"

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
            for folder in os.listdir(f"{self.config['data_dir']}/FAISS_store/"):
                if(os.path.isdir(f"{self.config['data_dir']}/FAISS_store/{folder}") and not(folder==temp_name)):
                    shutil.rmtree(f"{self.config['data_dir']}/FAISS_store/{folder}")
                    os.remove(f"{self.config['data_dir']}/FAISS_store/{folder}.faiss")

            # Rename temporary location
            os.rename(temp_dataset_path,f"{self.datastore_prefix}{self.kg_loaded_version}")
            os.rename(temp_faiss_path,f"{self.datastore_prefix}{self.kg_loaded_version}.faiss")

            self.kg2ds_map['datastore_version'] = self.kg_loaded_version
            self.kg2ds_map['unsaved_versions'] = [self.kg_loaded_version]
            with open(self.kg_to_ds_path, "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

            # Add timestamp to nodes
            for node in self.nodes_to_add:
                self.knowledge_graph.nodes[node[0]]['saved'] = int(time.time())

            self.reload_faiss_dataset()

            print("FAISS datastore successfully created and saved!")
        else:
            print("No new data to be saved!")

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

    def add_node(self,node,type,summary=""):

        print()
        # Add a node to the graph
        if(not(self.knowledge_graph.has_node(slugify_key(node)))):

            node_wiki = dict()
            node_wiki['summary'] = summary
            self.knowledge_graph.add_node(slugify_key(node),
                                            updated=int(time.time()),
                                            saved=-int(time.time()),
                                            name=node,
                                            type=type,
                                            wiki=node_wiki)
            if(summary):
                self.set_node_summary(node,summary)
        else:
            if(summary):
                self.set_node_summary(node,summary)
                self.knowledge_graph.nodes[node]['updated'] = int(time.time())
                
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

    # UPDATE THIS
    def get_nodes_to_add_to_faiss(self):
        # Return a list of nodes that have been updated, but not added to the FAISS dataset
        # Decompose a node into wiki keys and check if the ids are in the dataset.

        if(not(self.dataset)):
            return [node for node in self.knowledge_graph.nodes(data=True)]
        else:
            nodes_in_dataset = set(self.dataset['id'])

            print([node for node in self.knowledge_graph.nodes(data=True) if not(node[0] in nodes_in_dataset)])
            return [node for node in self.knowledge_graph.nodes(data=True) if not(node[0] in nodes_in_dataset)]

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
            self.knowledge_graph.nodes[node_id]['saved']    =  -int(time.time())

    def check_if_node_exists(self,node):
        return self.knowledge_graph.has_node(slugify_key(node))

    def set_node_summary(self,node,summary):

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['wiki']['summary'] = summary
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            print("Set Summary : ",self.knowledge_graph.nodes[node])
            return True

        return False

    def set_node_wiki_description(self,node_id,key,description):
        if self.knowledge_graph.has_node(node_id):
            self.knowledge_graph.nodes[node_id]['wiki'][key] = description
            self.knowledge_graph.nodes[node_id]['updated'] = int(time.time())

    def get_node_info(self,node):

        # All info is structured as {node_name, node_summary, list((node,tail,info))}

        node = slugify_key(node)
        if(self.knowledge_graph.has_node(node)):
            neighbors = list(self.knowledge_graph.adj[node])

            all_info = dict()

            all_info['id'] = node
            all_info['node_name'] = self.knowledge_graph.nodes[node]['name']
            all_info['node_type'] = self.knowledge_graph.nodes[node]['type']
            all_info['summary'] = "" # Remove this later
            all_info['wiki'] = self.knowledge_graph.nodes[node]['wiki']
            all_info['edge_info'] = []
            for neighbor in neighbors:
                all_info['edge_info'] += self.get_edge(node, neighbor)

            return all_info
        else:
            return None

    def get_node_id(self,node):
        return slugify_key(node)
        
    def get_node_summary(self,node):
        return self.knowledge_graph.nodes[node]['wiki']['summary'] if self.knowledge_graph.has_node(node) else None
    
    # GRAPH FUNCTIONS

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
            return any(sub_path == main_path[i : i + n] for i in range(len(main_path) - n + 1))
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

        if(not(nx.utils.misc.graphs_equal(self.knowledge_graph,most_recent_graph))):
            print("Saving Graph!")
            if(draw_figure):
                fig = plt.figure(figsize=(10, 10))
                pos = nx.spring_layout(G, seed=42) # Positions nodes cleanly

                # Draw nodes and edges
                nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
                nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=20)
                nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')

                # Draw edge labels (the snake_case relations)
                edge_labels = nx.get_edge_attributes(G, 'relation')
                nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

                plt.axis('off')
                plt.savefig(f'{config['graph_dir']}/assets/{name}')

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

    def get_graph_rag_context(self,query,threshold=0.4,k=10,hops=1):
        #def get_graph_rag_context(self,query,graph,documents_lookup,threshold=0.4,k=10,hops=1):

        # need self.knowledge_graph, self.documents_lookup

        print(query)

        query_vector = self.embed_text(query)
        scores, examples = self.dataset.get_nearest_examples("embeddings", query_vector, k=k)

        seed_results = []
        for i in range(len(scores)):
            if(scores[i] >= threshold and len(examples['text'][i].split(':')[-1].strip())>0):
                seed_results.append({feature:examples[feature][i] for feature in self.dataset.features.keys() if not(feature=='embeddings')})

        nodes_in_results = list(set([result['node_id'] for result in seed_results]))
        for result in seed_results:
            print(result)
            #print(f"{result['topic']} : {result['text']}")

        retrieved_context = []
        edge_paths = []

        if(len(nodes_in_results)>1):
            nodes = list(set([s['node_id'].strip() for s in seed_results]))
            print(nodes)

            # Get paths between all nodes first and resolve before extracting context
            node_pairs = [(nodes[idx],nodes[jdx]) for idx in range(1,len(nodes)) for jdx in range(idx) if not(idx==jdx)]

            # Ignore sub-paths
            all_paths = []
            for source,target in node_pairs:
                if nx.has_path(self.knowledge_graph, source, target):
                    all_paths.append(nx.shortest_path(self.knowledge_graph, source=source, target=target))

            all_paths = sorted(all_paths,key=len,reverse=True)
            print(all_paths)

            filtered_paths = []
            for path in all_paths:
                if(not(any([any([self.check_subpath(path,c_path),self.check_subpath(path[::-1],c_path)]) for c_path in filtered_paths]))):
                    filtered_paths.append(path)
            filtered_paths = sorted(filtered_paths,key=len,reverse=True)
            print(filtered_paths)

            for path in filtered_paths:

                head,tail = path[0],path[-1]
                knowledge_chain = [self.get_node_summary(head)]

                if(len(path)>1):
                    for path_idx in range(len(path)-1):
                        head,tail = path[path_idx:path_idx+2]
                        edge_paths+=[(head,tail,key,edge_dict['desc']) for key,edge_dict in self.knowledge_graph.get_edge_data(head,tail).items()]
                        knowledge_chain += [edge_data['desc'].strip() for edge_data in self.knowledge_graph.get_edge_data(head,tail).values()]

                if(not(head==tail)):
                    knowledge_chain += [self.get_node_summary(tail)]
                print(knowledge_chain)
                retrieved_context.append(' '.join(knowledge_chain))   
        elif(len(nodes_in_results)==0):
            retrieved_context = []
            edge_paths = [] 
        else:
            # If there is only one node in the results
            node = seed_results[0]['node_id'].strip()
            retrieved_context = [self.knowledge_graph.nodes[node]['wiki']['summary']]
            retrieved_context += list(set([result['text'] for result in seed_results]))
            edge_paths += [(r['node_id'],r['node_id'],r['tag'],r['text']) for r in seed_results]
            
        return retrieved_context, edge_paths

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
