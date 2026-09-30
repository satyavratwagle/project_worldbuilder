import torch
from gliner import GLiNER
import utils.preprocessing as pre
import numpy as np
from sentence_transformers import SentenceTransformer
import datasets
import networkx as nx
from datasets import Dataset,concatenate_datasets
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

    def __init__(self,config,project_name="worldbuilding",current_graph='lore',is_core_graph=False,load_most_recent=True):

        self.config = config
        self.kg_to_ds_path = config['kg_to_ds_map']

        self.project_name = project_name

        # Create the project folders in graphs and datasets if they do not exist.
        os.makedirs(f"{self.config['graph_dir']}/{self.project_name}", exist_ok=True)
        os.makedirs(f"{self.config['datastore_dir']}/{self.project_name}", exist_ok=True)
        
        self.current_graph = current_graph
        self.knowledge_graph_prefix = f"{self.config['graph_dir']}/{self.project_name}/{self.current_graph}_"
        self.datastore_prefix = f"{self.config['datastore_dir']}/{self.project_name}/{self.current_graph}_"

        # Load knowledge-graph to datastore map

        if(not(os.path.exists(self.kg_to_ds_path))):
            # kg_to_ds_path should indicate the knowledge graph version contained in the current FAISS dataset and the ones not tracked
            kg2ds = dict()
            kg2ds[self.project_name] = dict()
            kg2ds[self.project_name][self.current_graph] = dict()
            kg2ds[self.project_name][self.current_graph]['datastore_version'] = None
            kg2ds[self.project_name][self.current_graph]['datastore_last_saved'] = 0
            kg2ds[self.project_name][self.current_graph]['unsaved_versions'] = []
            kg2ds[self.project_name][self.current_graph]['is_core_graph'] = is_core_graph
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
                self.kg2ds_map[self.project_name][self.current_graph]['datastore_version'] = None
                self.kg2ds_map[self.project_name][self.current_graph]['datastore_last_saved'] = 0
                self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'] = []
                self.kg2ds_map[self.project_name][self.current_graph]['is_core_graph'] = is_core_graph

        # First check for the core knowledge graph

        self.core_graph = None
        if(not(self.kg2ds_map[self.project_name][self.current_graph]['is_core_graph'])):
            for key in self.kg2ds_map[self.project_name].keys():
                if(self.kg2ds_map[self.project_name][key]['is_core_graph']):
                    self.core_graph_name = key

                    self.core_knowledge_graph_prefix = f"{self.config['graph_dir']}/{self.project_name}/{self.core_graph_name}_"
                    self.core_datastore_prefix = f"{self.config['datastore_dir']}/{self.project_name}/{self.core_graph_name}_"

                    print(f"Core Graph for project {self.project_name} : {self.core_knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.core_graph_name]['unsaved_versions']}.gml")
                    if(len(self.kg2ds_map[self.project_name][key]['unsaved_versions'])>0 and os.path.exists(f"{self.core_knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.core_graph_name]['unsaved_versions'][-1]}.gml")):
                        self.core_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.core_graph_name]['unsaved_versions'][-1]}.gml")

            if(not(self.core_graph)):
                # If core graph does not exist, create an empty graph to serve as a core graph to make things cleaner later.
                self.core_graph = nx.MultiGraph()
            

        # Load up the current graph
        
        if(len(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")):
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")
            self.edge_key = (max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])+1) if len(self.knowledge_graph.edges(keys=True))>0 else 1
            self.kg_loaded_version = self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]
        else:
            kg_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.knowledge_graph = nx.MultiGraph()
            nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_prefix}{kg_suffix}.gml")
            self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'].append(kg_suffix)
            self.edge_key = 1
            self.kg_loaded_version = kg_suffix

        print(f"Loaded Graph Version : {self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]} : {self.knowledge_graph}")

        if(self.kg2ds_map[self.project_name][self.current_graph]['is_core_graph']):

            print('Setting Core Graph as Current Graph!')

            self.core_graph = self.knowledge_graph
            self.core_graph_name = self.current_graph
            print("Core Graph is the current graph!")
            
            self.core_knowledge_graph_prefix = self.knowledge_graph_prefix
            self.core_datastore_prefix = self.datastore_prefix

        assert not(self.core_graph==None)

        # Load up the datastore for the core graph

        if(not(self.core_graph_name == self.current_graph)):

            if(self.kg2ds_map[self.project_name][self.core_graph_name]['datastore_version'] and not(os.path.exists(f"{self.core_datastore_prefix}{self.kg_loaded_version}"))):
                self.kg2ds_map[self.project_name][self.core_graph_name]['datastore_version'] = None

            if(self.kg2ds_map[self.project_name][self.core_graph_name]['datastore_version']):
                self.core_dataset = datasets.load_from_disk(f"{self.core_datastore_prefix}{self.kg_loaded_version}")
                self.core_dataset.load_faiss_index("embeddings", f"{self.core_datastore_prefix}{self.kg_loaded_version}.faiss")
            else:
                # No datastore was found
                self.core_dataset = None
        
        # Load up the datastore for the current graph

        if(self.kg2ds_map[self.project_name][self.current_graph]['datastore_version'] and not(os.path.exists(f"{self.datastore_prefix}{self.kg_loaded_version}"))):
            self.kg2ds_map[self.project_name][self.current_graph]['datastore_version'] = None

        if(self.kg2ds_map[self.project_name][self.current_graph]['datastore_version']):
            self.dataset = datasets.load_from_disk(f"{self.datastore_prefix}{self.kg_loaded_version}")
            self.dataset.load_faiss_index("embeddings", f"{self.datastore_prefix}{self.kg_loaded_version}.faiss")
        else:
            # No datastore was found
            self.dataset = None

        # Update the kg_to_ds_map
        with open(config['kg_to_ds_map'], "w") as f:
            json.dump(self.kg2ds_map, f, indent=4)

        # Find nodes based on aliases and map them back to the node

        self.all_aliases = [alias for node in self.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']]
        print("Aliases : ",self.all_aliases)
        alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias in self.all_aliases)})\b"
        self.node_finder = re.compile(alias_pattern, flags=re.IGNORECASE)
        self.alias_map = {alias:node[0] for node in self.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']}

        print("Alias Map : ",self.alias_map)
            
    # FAISS DATASET FUNCTIONS

    def create_dataset_from_nodes(self,nodes,get_index=False):
        # Create a dataset from a set of nodes 
        # Use get_index if generating dataset for the first time.

        documents = []
        for node in nodes:
            update_time = int(time.time())
            for key,statement in self.get_node_summary(node[0]).items():
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
        print('filter_dataset')

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
            temp_dataset_path = f"{self.config['datastore_dir']}/{self.project_name}/{temp_name}"
            temp_faiss_path = f"{self.config['datastore_dir']}/{self.project_name}/{temp_name}.faiss"

            new_dataset.info.description = str(int())

            # Save to temporary location
            new_dataset.save_to_disk(temp_dataset_path)
            new_dataset.add_faiss_index(column="embeddings", custom_index=cosine_index, index_name="faiss_cosine")

            if os.path.exists(temp_faiss_path):
                os.remove(temp_faiss_path)
            new_dataset.save_faiss_index("faiss_cosine", temp_faiss_path)

            # Delete old dataset if exists
            for folder in os.listdir(f"{self.config['datastore_dir']}/{self.project_name}"):
                if(os.path.isdir(f"{self.config['datastore_dir']}/{self.project_name}/{folder}") and not(folder==temp_name)):
                    shutil.rmtree(f"{self.config['datastore_dir']}/{self.project_name}/{folder}")
                    os.remove(f"{self.config['datastore_dir']}/{self.project_name}/{folder}.faiss")

            # Rename temporary location
            os.rename(temp_dataset_path,f"{self.datastore_prefix}{self.kg_loaded_version}")
            os.rename(temp_faiss_path,f"{self.datastore_prefix}{self.kg_loaded_version}.faiss")

            self.kg2ds_map[self.project_name][self.current_graph]['datastore_version'] = self.kg_loaded_version
            self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'] = [self.kg_loaded_version]

            # Keep only the latest graph
            for graph in os.listdir(f"{self.config['graph_dir']}/{self.project_name}/"):
                if(not(graph==f"{self.current_graph}_{self.kg_loaded_version}.gml")):
                    os.remove(f"{self.config['graph_dir']}/{self.project_name}/{graph}")

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

        if(self.kg2ds_map[self.project_name][self.current_graph]['datastore_version'] and os.path.exists(f"{self.datastore_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['datastore_version']}")):
            self.dataset = datasets.load_from_disk(f"{self.datastore_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['datastore_version']}")
            self.dataset.load_faiss_index("embeddings", f"{self.datastore_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['datastore_version']}.faiss")
        else:
            self.dataset = None
        print(self.dataset)

    # EDGE FUNCTIONS

    def add_edge(self,head,tail,text,edge_label=""):
        # Add an edge to the graph
        time_now = int(time.time())
        self.knowledge_graph.add_edge(slugify_key(head), 
                                        slugify_key(tail),
                                        desc=text,
                                        key=str(self.edge_key),
                                        label=edge_label.lower(), 
                                        updated=time_now)
        self.edge_key += 1

        return str(self.edge_key-1)

    def remove_edge(self,head,tail,key):
        if(self.knowledge_graph.has_edge(head,tail,key=key)):
            self.knowledge_graph.remove_edge(head,tail,key=key)

    def get_edges_from(self,head):
        return [edge for edge in self.knowledge_graph.edges(head,keys=True,data=True)]

    def get_edge(self,head,tail):
        # Get all edges between a head and tail node
        return [(head,tail,k,v['desc'],v['label'],[self.get_node_name(head),self.get_node_name(tail)]) for k,v in self.knowledge_graph.adj[head][tail].items()]

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

    def set_edge_label(self,edge,topic,label):
        head,tail,key,metadata = edge
        self.knowledge_graph.edges[head,tail,key]['label'] = label
        self.knowledge_graph.edges[head,tail,key]['updated'] = int(time.time())

    def get_num_edges(self):
        # Return the number of edges in the graph
        return max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])

    def resolve_edges(self,threshold=0.9):
        # Remove redundant edges and add new connections based on edge data if any
        
        # Remove Redandant Edges
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

    def resolve_text(self,text_list,threshold=0.95):

        similar_list = []
        idxs_to_keep = []
        embeddings = torch.from_numpy(self.text_embedding_model.encode(text_list, normalize_embeddings=True))
        cosine_sim = torch.matmul(embeddings,embeddings.T)

        for idx,text in enumerate(text_list):
            print('Similarity Check - ',text," : ",cosine_sim[idx])

        for idx in range(len(text_list)):
            if(not(idx in similar_list)):
                similar_idxs = torch.nonzero(cosine_sim[idx]>threshold,as_tuple=True)[0]
                similar_list += [sim_idx for sim_idx in similar_idxs if not(sim_idx in similar_list)]
                idxs_to_keep.append(idx)

        return idxs_to_keep

    def find_new_edges(self):

        # Check for any new connections

        edges_log = []

        edges_to_add = []
        edges_to_remove = []
        for edge in self.knowledge_graph.edges(keys=True,data=True):

            old_head = edge[0]
            old_tail = edge[1]
            old_key = edge[2]
            desc = edge[3]['desc']
            label = edge[3]['label']

            relevant_nodes = [self.alias_map[name] for name in list(set(self.node_finder.findall(desc))) if len(name)>0]

            if(len(relevant_nodes)>1):
                pairs = [(relevant_nodes[jdx],relevant_nodes[idx]) for idx in range(1,len(relevant_nodes)) for jdx in range(idx)]
                
                for head,tail in pairs:
                    already_added = False
                    if(tail in self.knowledge_graph.neighbors(head)):
                        edge_keys = [edge[2] for edge in self.knowledge_graph.edges([head,tail],keys=True,data=True)]
                        if(old_key in edge_keys):
                            already_added = True

                    if(not(already_added)):
                        edges_to_add.append((head,tail,desc,label))
                        edges_to_remove.append((old_head,old_tail,old_key))
                        edges_log.append((self.knowledge_graph.nodes[head]['name'],self.knowledge_graph.nodes[tail]['name'],desc))
                        print(f"Added edge '{desc}' between {head} and {tail}")

        for head,tail,desc,label in edges_to_add:
            self.add_edge(head,tail,desc,edge_label=label)

        for old_head,old_tail,old_key in edges_to_remove:
            self.remove_edge(old_head,old_tail,old_key)

        return edges_log

    def format_edge_data(self,head,tail,key):

        return f"- {self.knowledge_graph.edges[head,tail,key]["label"]} : {self.knowledge_graph.edges[head,tail,key]["desc"]}"

    # NODE FUNCTIONS

    def add_node(self,node,type,aliases=[],summary=None,definition=""):
        # NOTE : Aliases needed for relevant node retrieval

        node_name = node
        node = slugify_key(node)

        if(summary):
            assert type(summary)==dict
        else:
            summary = dict()
            summary['lore'] = 'No Lore-related information available'
            summary['plot'] = 'No Plot-related information available'

        # Add a node to the graph
        if(not(self.knowledge_graph.has_node(node))):

            self.knowledge_graph.add_node(node,
                                            updated=int(time.time()),
                                            name=node_name,
                                            type=type,
                                            summary=summary,
                                            aliases=aliases,
                                            definition="")
        else:
            if(len(aliases)>0):
                self.knowledge_graph.nodes[node]['aliases'] += aliases
                self.knowledge_graph.nodes[node]['aliases'] = list(set(self.knowledge_graph.nodes[node]['aliases']))

        if(len(summary.keys())>0):
            self.knowledge_graph.nodes[node]['summary'] = summary

        return slugify_key(node)
                
    def get_node_name(self,node_id):
        return self.knowledge_graph.nodes[node_id]['name']

    def get_node_id(self,node_name):

        for node in self.knowledge_graph.nodes(data=True):
            if(node[1]['name']==node_name):
                return node[0]

        return None

    def remove_node(self,node):
        # Cleanly remove a node from the knowledge graph

        node = slugify_key(node)

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

    # GRAPH FUNCTIONS
    def check_if_node_exists(self,node):
        return self.knowledge_graph.has_node(slugify_key(node))

    def has_new_edges(self,node):

        needs_update = any([edge[3]['updated']>=self.knowledge_graph.nodes[node]['updated'] for edge in self.get_edges_from(node)])

        return needs_update

    def cluster_edge_info(self,node):

        print('cluster_edge_info')
        node = slugify_key(node)

        all_edge_desc   = [edge[3]['desc'] for edge in self.get_edges_from(node)]
        all_edge_labels = list(set([edge[3]['label'] for edge in self.get_edges_from(node)]))
        print([edge[3]['label'] for edge in self.get_edges_from(node)])

        edge_clusters = dict()
        for label in all_edge_labels:

            info_list = [edge[3]['desc'] for edge in self.get_edges_from(node) if edge[3]['label']==label]
            if(len(info_list)>0):
                print(label)
                edge_clusters[uppercase(label)] = ' '.join(info_list)

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

    def get_node_definition(self,node):
        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            return self.knowledge_graph.nodes[node]['definition']

    def reset_node_summary(self,node):

        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['summary'] = dict()
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            return True

        return False

    def add_to_node_summary(self,node,topic,summary):

        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['summary'][topic] = summary
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            print(f"Set Summary of {node} : ",self.knowledge_graph.nodes[node]['summary'])
            return True

        return False
    
    def get_node_summary(self,node):
        return self.knowledge_graph.nodes[slugify_key(node)]['summary'] if self.knowledge_graph.has_node(slugify_key(node)) else dict()
    
    def format_node_data(self,node,edge_filters=None):

        node = slugify_key(node)

        if self.knowledge_graph.has_node(node):
            node_summary = self.knowledge_graph.nodes[node]['summary']

        formatted_summary = ''
        for key,value in node_summary.items():

            if(edge_filters):
                if(key in edge_filters):
                    formatted_summary += f"- {key} : {value}\n"
            else:
                formatted_summary += f"- {key} : {value}\n"

        return formatted_summary

    def explore_neighborhood(self,node,neighborhood=[],n_hops=1,edge_filters=None):

        if(edge_filters):
            def filter_function(u, v, k):
                return self.knowledge_graph.edges[u,v,k]["label"] in edge_filters

            neighborhood_graph = nx.subgraph_view(self.knowledge_graph, filter_edge=filter_function)
        else:
            neighborhood_graph = self.knowledge_graph

        node = slugify_key(node)

        if(n_hops>0):
            node_neighbors = self.knowledge_graph.neighbors(node)
            for neighbor in node_neighbors:
                if(not((node,neighbor) in neighborhood) and not((neighbor,node) in neighborhood)):
                    neighborhood.append((node,neighbor))
                    new_set = self.explore_neighborhood(neighbor,n_hops=(n_hops-1),neighborhood=neighborhood,edge_filters=edge_filters)

        return neighborhood

    def find_path_between(self,head,tail,edge_filters=None):

        source = slugify_key(source)
        target = slugify_key(target)

        path = None

        if(self.knowledge_graph.has_node(source) and self.knowledge_graph.has_node(target)):
            if(edge_filters):

                def filter_function(u, v, k):
                    return self.knowledge_graph.edges[u,v,k]["label"] in edge_filters

                filtered_graph = nx.subgraph_view(self.knowledge_graph, filter_edge=filter_function)

                if nx.has_path(filtered_graph, source, target):
                    path = nx.shortest_path(filtered_graph, source=source, target=target)
            else:
                if nx.has_path(self.knowledge_graph, source, target):
                    path = nx.shortest_path(self.knowledge_graph, source=source, target=target)

        return path

    def get_relevant_nodes(self,query,k=10, threshold=0.4):

        # Check literal mentions of any node names.
        relevant_nodes = [self.alias_map[name] for name in list(set(self.node_finder.findall(query))) if len(name)>0]
        print("Extracted Nodes :",relevant_nodes)

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

    def find_all_unique_paths(self,nodes, edge_filters=None):

        if(edge_filters):
            def filter_function(u, v, k):
                return self.knowledge_graph.edges[u,v,k]["label"] in edge_filters

            filtered_graph = nx.subgraph_view(self.knowledge_graph, filter_edge=filter_function)
        else:
            filtered_graph = self.knowledge_graph


        all_paths = []
        for idx in range(len(nodes)):
            for jdx in range(len(nodes)):
                if(not(idx==jdx)):
                    path = self.find_path(nodes[idx],nodes[jdx],filtered_graph=filtered_graph)
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

    def get_unparsed_edges(self,node):
        unparsed_edges = [edge for edge in self.knowledge_graph.edges(node,keys=True,data=True) if self.knowledge_graph.nodes[node]['updated']<edge[3]['updated']]
        return unparsed_edges

    def get_node_summaries(self,nodes):
        return [self.get_node_summary(node) for node in nodes]

    def check_nodes_for_replacement(self,named_entities,threshold=0.85):
        nodes = self.get_all_nodes()

        similar_pairs = []
        for key in named_entities:
            similarity = list(map(lambda x :difflib.SequenceMatcher(None, x,key).ratio(),self.all_aliases))
            similar_pairs += [(key,self.knowledge_graph.nodes[self.alias_map[self.all_aliases[i]]]) for i, sim in enumerate(similarity) if (sim > threshold)]
        
        return similar_pairs

    def find_path(self,source,target,filtered_graph=None):

        source = slugify_key(source)
        target = slugify_key(target)

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
            edge_data['text'] = edge[3]['desc']
            edges.append(edge_data)
        
        nodes = dict()
        for node in self.knowledge_graph.nodes(data=True):
            
            nodes[node[0]] = dict()
            nodes[node[0]]['name'] = node[1]['name']
            nodes[node[0]]['type'] = node[1]['type']

        graph_data['nodes'] = nodes
        graph_data['edges'] = edges

        with open(f"{self.config['data_dir']}/graph_json_{self.kg2ds_map['unsaved_versions'][-1]}.json", "w") as f:
            json.dump(graph_data, f, indent=4)

    def save_graph(self,draw_figure=False):

        # Reload kg_to_ds_map
        with open(self.kg_to_ds_path, "r", encoding="utf-8") as f:
            self.kg2ds_map = json.load(f)

        if(len(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'])>0):
            # Check if graphs exist. If not, remove them from the map
            versions_to_remove = []
            for idx,kg_version in enumerate(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions']):
                if(not(os.path.exists(f"{self.knowledge_graph_prefix}{kg_version}.gml"))):
                    versions_to_remove.append(kg_version)
            [self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'].remove(version) for version in versions_to_remove]

            with open(self.config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

        if(len(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")):
            # load the latest knowledge graph
            most_recent_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")
        else:
            most_recent_graph = None

        if(True):
            print("Saving Graph!")
            # Clean up old graphs
            # 1 day = 86400 seconds

            self.kg_saved_version = datetime.now().strftime("%Y%m%d_%H%M%S")
            nx.write_gml(self.knowledge_graph, f"{self.knowledge_graph_prefix}{self.kg_saved_version}.gml")

            self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'].append(self.kg_saved_version)

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

        if(len(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'])>0):
            # Check if graphs exist. If not, remove them from the map
            versions_to_remove = []
            for idx,kg_version in enumerate(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions']):
                if(not(os.path.exists(f"{self.knowledge_graph_prefix}{kg_version}.gml"))):
                    versions_to_remove.append(kg_version)
            [self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'].remove(version) for version in versions_to_remove]

            with open(self.config['kg_to_ds_map'], "w") as f:
                json.dump(self.kg2ds_map, f, indent=4)

        if(len(self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'])>0 and os.path.exists(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")):
            # load the latest knowledge graph
            self.knowledge_graph = nx.read_gml(f"{self.knowledge_graph_prefix}{self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]}.gml")
            self.edge_key = (max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])+1) if len(self.knowledge_graph.edges(keys=True))>0 else 1
            self.kg_loaded_version = self.kg2ds_map[self.project_name][self.current_graph]['unsaved_versions'][-1]
            print(f"Reloaded Graph Version : {self.kg_loaded_version} : {self.knowledge_graph}")
            self.all_aliases = [alias for node in self.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']]
            self.alias_map = {alias:node[0] for node in self.knowledge_graph.nodes(data=True) for alias in node[1]['aliases']}
            alias_pattern = rf"\b({'|'.join([re.escape(alias) for alias in self.all_aliases])})\b"
            self.node_finder = re.compile(alias_pattern, flags=re.IGNORECASE)
            return True
        else:
            return False

    def load_embedding_model(self):
        self.text_embedding_model = SentenceTransformer(self.config['text_embedding_model'])

    # RAG FUNCTIONS

    def get_context_from_paths(self,paths,edge_filters=None):

        path_data_to_format = []
        nodes_added = []
        edges_added = []
        context = ""
        for path in paths:
            
            for idx in range(len(path)-1):
                head = path[idx]
                tail = path[idx+1]

                if(not(head in nodes_added)):
                    path_data_to_format.append(('node',self.format_node_data(head,None),self.get_node_name(head)))
                    nodes_added.append(head)

                relevant_edges = []
                for edge in self.get_edge(head,tail):
                    head,tail,key,text,label,endpoints = edge
                    if(edge_filters):
                        if(label in edge_filters):
                            relevant_edges.append(self.format_edge_data(head,tail,key))
                            edges_added.append((head,tail,key))
                    else:
                        relevant_edges.append(self.format_edge_data(head,tail,key))
                        edges_added.append((head,tail,key))

                path_data_to_format.append(('edge','\n'.join(relevant_edges),self.get_node_name(head),self.get_node_name(tail)))


                if(not(tail in nodes_added)):
                    path_data_to_format.append(('node',self.format_node_data(tail,None),self.get_node_name(tail)))
                    nodes_added.append(tail)

            context += self.format_context(path_data_to_format)+'\n\n'

        return context, nodes_added, edges_added

    def get_context_from_neighborhood(self,neighborhood,edge_filters=None):

        if(edge_filters):
            def filter_function(u, v, k):
                return self.knowledge_graph.edges[u,v,k]["label"] in edge_filters

            filtered_graph = nx.subgraph_view(self.knowledge_graph, filter_edge=filter_function)
        else:
            filtered_graph = self.knowledge_graph

        relevant_edges = []
        for head,tail in neighborhood:
            relevant_edges += [(head,tail,k) for k,v in filtered_graph.adj[head][tail].items()]

        relevant_edges = list(set(relevant_edges))

        relevant_nodes = list(set([r[0] for r in relevant_edges]+[r[1] for r in relevant_edges]))

        nodes_to_format = []
        for node in relevant_nodes:
            # TODO
            nodes_to_format.append(('node',self.format_node_data(node,edge_filters),self.get_node_name(node)))

        formatted_node_context = self.format_context(nodes_to_format)

        edges_to_format = []
        added_edges = []
        for edge in relevant_edges:
            head,tail,key = edge
            edges_to_format.append(('edge',self.format_edge_data(head,tail,key),self.get_node_name(head),self.get_node_name(tail)))
            added_edges.append((head,tail,key))

        formatted_edge_context = self.format_context(edges_to_format)

        formatted_context = f"{formatted_node_context}\n{formatted_edge_context}"

        return formatted_context, relevant_nodes, added_edges

    def get_graph_rag_context(self,user_query,strategy='explore_neighborhood',categories=None,threshold=0.8,k=10,n_hops=1):

        nodes_in_results = self.get_relevant_nodes(user_query,threshold=threshold,k=k)
        nodes = None
        edges = None

        if(strategy=='explore_neighborhood'):
            neighborhood = []
            for node in nodes_in_results:
                neighborhood = self.explore_neighborhood(node,neighborhood,n_hops=n_hops,edge_filters=categories)

            context, nodes, edges = self.get_context_from_neighborhood(neighborhood,edge_filters=categories)

        if(strategy=='find_path_between'):

            if(categories):
                def filter_function(u, v, k):
                    return self.knowledge_graph.edges[u,v,k]["label"] in categories

                filtered_graph = nx.subgraph_view(self.knowledge_graph, filter_edge=filter_function)
            else:
                filtered_graph = self.knowledge_graph

            edge_paths = self.find_all_unique_paths(nodes_in_results, edge_filters=categories)

            context, nodes, edges = self.get_context_from_paths(edge_paths,edge_filters=categories)
        return context, nodes, edges

    def legacy_get_graph_rag_context(self,query,threshold=0.4,k=10):
        #def get_graph_rag_context(self,query,graph,documents_lookup,threshold=0.4,k=10,hops=1):

        # need self.knowledge_graph, self.documents_lookup

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

    def format_context(self,paths):
        context_text = ""
        for p in paths:
            if(p[0]=='node'):
                context_text += f"<information>\nTopic : {p[2]}\n{p[1]}</information>\n\n"
            if(p[0]=='edge'):
                if(p[2]==p[3]):
                    context_text += f"<information>\nTopic : {p[2]}\n{p[1]}\n</information>\n\n"
                else:
                    context_text += f"<relationship>\nTopic : Relationship between {p[2]} and {p[3]}\n{p[1]}\n</relationship>\n\n"

        return context_text

    def create_network_figure(self, target_nodes, target_edges=None):
        # target_nodes : list(node_ids)
        # target_edges : list((head,tail,key))

        # Create a subgraph with the target nodes and target edges
        all_neighbors = []
        for node in target_nodes:
            all_neighbors += self.knowledge_graph.neighbors(node)

        all_neighbors = list(set(all_neighbors))

        new_edges = []
        for node in target_nodes:
            for neighbor in all_neighbors:
                if(neighbor in self.knowledge_graph.neighbors(node)):
                    new_edges += [(node,neighbor,k) for k in self.knowledge_graph.adj[node][neighbor].keys()]
        new_edges = list(set(new_edges))

        G = self.knowledge_graph.subgraph(target_nodes+all_neighbors).copy()
        # 2. Extract the edge-induced subgraph
        if(target_edges):
            G = G.edge_subgraph(target_edges+new_edges)

        pos = nx.spring_layout(G, seed=42)

        # 2. Extract unique edges and aggregate data
        edge_x, edge_y = [], []
        mid_x, mid_y, edge_text = [], [], []
        endpoints = []
        seen_edges = set()

        other_edge_x, other_edge_y = [], []

        for u, v, data in G.edges(data=True):
            # Sort tuple for undirected graphs to avoid handling (A,B) and (B,A) separately
            edge_key = tuple(sorted([u, v]))
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)

                endpoints.append((u,v))

                if(u in target_nodes and v in target_nodes):
                    x0, y0 = pos[u]
                    x1, y1 = pos[v]
                    
                    if(not(u==v)):
                        edges = G.adj[u][v]
                        pair_text = [e['desc'] for e in edges.values()]
                    else:
                        pair_text = []

                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])


                    # Midpoint coordinates for edge hover triggering
                    if(not(u==v)):
                        mid_x.append((x0 + x1) / 2)
                        mid_y.append((y0 + y1) / 2)
                        edge_text.append("<br>".join([text for text in pair_text]))
                else:

                    x0, y0 = pos[u]
                    x1, y1 = pos[v]

                    other_edge_x.extend([x0, x1, None])
                    other_edge_y.extend([y0, y1, None])


        edge_trace = go.Scatter(
          x=edge_x,
          y=edge_y,
          line=dict(width=5.0, color="#EEE"),
          hoverinfo="none",
          mode="lines",
        )

        other_edge_trace = go.Scatter(
          x=other_edge_x,
          y=other_edge_y,
          line=dict(width=2.0, color="#666"),
          hoverinfo="none",
          mode="lines",
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
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)

            wrapped_desc = "<br>".join(textwrap.wrap(node[1]['summary']['lore'], width=35))
            node_text.append(f"<b>{node[1]['name']}</b><br><i>{wrapped_desc}</i>")
            node_ids.append(node[0])
            if(node[0] in target_nodes):
                node_colors.append("#EEE")
                node_labels.append(self.get_node_name(node[0]))
            else:
                node_colors.append("#333")
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
                color="#EEE",  # Color of the text labels (e.g., white)
                size=20,  # Font size in pixels
                family=('system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",'' Roboto, Helvetica, Arial, sans-serif')
            ),
          marker=dict(
              showscale=False,
              color="#EEE",
              size=30)
        )

        # Zoom into the parts of the graph that are relevant                
        node_trace.marker.color = node_colors
        node_trace.text = node_labels


        # 2. Extract coordinates for just these nodes
        subset_x = [pos[node][0] for node in target_nodes if node in pos]
        subset_y = [pos[node][1] for node in target_nodes if node in pos]

        # 3. Calculate bounding box ranges with a 20% padding margin
        if subset_x and subset_y:
          x_min, x_max = min(subset_x), max(subset_x)
          y_min, y_max = min(subset_y), max(subset_y)

          x_pad = (x_max - x_min) * 0.2 if x_max != x_min else 1.0
          y_pad = (y_max - y_min) * 0.2 if y_max != y_min else 1.0

          x_range = [x_min - x_pad, x_max + x_pad]
          y_range = [y_min - y_pad, y_max + y_pad]
        else:
          # Fallback defaults if target nodes aren't found
          x_range, y_range = None, None

        # 5. Build the final Plotly figure
        fig = go.Figure(
          data=[edge_trace, other_edge_trace, edge_hover_trace,node_trace],
          layout=go.Layout(
              showlegend=False,
              hovermode="closest",
              plot_bgcolor="rgba(0,0,0,0)",  # Transparent plot canvas
              paper_bgcolor="rgba(0,0,0,0)",  # Transparent outer border canvas
              hoverlabel=dict(
                    bgcolor="#ffffff",  # Background color of the tooltip bubble (e.g., dark gray)
                    font_color="#000000",  # Text color inside the tooltip
                    font_size=12,  # Font size
                    font_family=(
                        'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",'
                        ' Roboto, Helvetica, Arial, sans-serif'
                    ), # Font family
                    bordercolor="#ffffff",  # Border color around the bubble
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
        return fig

    