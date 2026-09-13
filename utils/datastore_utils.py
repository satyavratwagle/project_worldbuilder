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
import matplotlib.pyplot as plt
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

class DatastoreUtilities():

    def __init__(self,config):

        self.config = config
        self.store_path = config['data_dir']
        self.jsonstore_dir = f'{self.store_path}/json_store'
        self.faiss_dataset_path = f'{self.store_path}/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = f"{self.store_path}/FAISS_store/worldbuilding_dataset.faiss"
        self.knowledge_graph_path = f"{self.config['graph_dir']}/lore_graph.gml"

        if(os.path.exists(self.faiss_dataset_path)):
            self.dataset = datasets.load_from_disk(self.faiss_dataset_path)
            self.dataset.load_faiss_index("embeddings", self.faiss_index_path)
        else:
            self.dataset = None

        if(not(os.path.exists(self.knowledge_graph_path))):
            nx.write_gml(nx.MultiGraph(), self.knowledge_graph_path)
            self.knowledge_graph = nx.read_gml(self.knowledge_graph_path)
            self.edge_key = 1
        else:
            self.knowledge_graph = nx.read_gml(self.knowledge_graph_path)
            self.edge_key = max([int(e[2]) for e in self.knowledge_graph.edges(keys=True)])+1
            
    # FAISS DATASET FUNCTIONS

    def create_dataset_from_nodes(self,nodes,get_index=False):
        # Create a dataset from a set of nodes 
        # Use get_index if generating dataset for the first time.

        documents = []
        for node in nodes:
            documents.append({'topic':node,
                'text':self.knowledge_graph.nodes[node]['summary']})

        graph_dataset = Dataset.from_list(documents)

        cosine_index = None
        if(get_index):
            graph_dataset,cosine_index = self.calculate_faiss_index(graph_dataset)

        return graph_dataset,cosine_index

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

        # Add timestamp to nodes
        for node in self.nodes_to_add:
            self.knowledge_graph.nodes[node]['added'] = int(time.time())

        self.reload_faiss_dataset()

        print("FAISS datastore successfully created and saved!")

    def reload_faiss_dataset(self):

        if not(os.path.exists(self.faiss_dataset_path)):
            dataset,index = self.update_faiss_dataset()
            self.overwrite_faiss_dataset(dataset,index)
        else:
            self.dataset = datasets.load_from_disk(os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset"))
            self.dataset.load_faiss_index("embeddings", os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset.faiss"))

    def calculate_faiss_index(self,dataset):

        def embed_text(batch):
                # Return a dictionary mapping to your target index string key
                return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}
                
        embedding_dim = self.text_embedding_model.get_embedding_dimension()
        cosine_index = faiss.IndexFlatIP(embedding_dim)
        dataset = dataset.map(embed_text, batched=True, batch_size=8)

        return dataset, cosine_index

    def update_faiss_dataset(self):

        self.nodes_to_add = self.get_nodes_to_add_to_faiss()
        new_dataset,_ = self.create_dataset_from_nodes(self.nodes_to_add)

        print([doc['topic'] for doc in new_dataset])
        
        if(os.path.exists(self.faiss_dataset_path)):
            # Filter out existing data from old dataset
            old_dataset = datasets.load_from_disk(self.faiss_dataset_path)

            # Keep the new updated dataset values
            filtered_dataset = old_dataset
            filtered = False
            for doc in new_dataset:
                filtered_dataset = filtered_dataset.filter(lambda example: example["topic"] != doc['topic'])
                filtered = True

            # Remove any deleted files
            old_topics = list(set([doc['topic'] for doc in old_dataset]))
            excluded_topics = []
            for topic in old_topics:
                if not(self.knowledge_graph.has_node(topic)):
                    excluded_topics.append(topic)
                
            for excluded_topic in excluded_topics:
                filtered_dataset = filtered_dataset.filter(lambda example: example["topic"] != excluded_topic)
                
            # Append datasets.
            updated_dataset = concatenate_datasets([filtered_dataset,new_dataset])
        else:
            updated_dataset = new_dataset

        updated_dataset,updated_index = self.calculate_faiss_index(updated_dataset)

        return updated_dataset,updated_index

        #self.overwrite_faiss_dataset(updated_dataset,index)
        #self.reload_faiss_dataset()
    
    # GRAPH FUNCTIONS

    def add_edge(self,head,tail,text):
        # Add an edge to the graph
        self.knowledge_graph.add_edge(head, tail,desc=text,parsed=False,key=str(self.edge_key))
        self.edge_key += 1

        return str(self.edge_key-1)

    def remove_edge(self,head,tail,key):
        if(self.knowledge_graph.has_edge(head,tail,key=key)):
            self.knowledge_graph.remove_edge(head,tail,key=key)

    def add_node(self,node,summary=None):
        # Add a node to the graph
        if(not(self.knowledge_graph.has_node(node))):
            self.knowledge_graph.add_node(node,updated=int(time.time()),added=-int(time.time()),summary="")
            if(summary):
                self.set_node_summary(node,summary)
        else:
            if(summary):
                self.set_node_summary(node,summary)
                self.knowledge_graph.nodes[node]['updated'] = int(time.time())
                
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
        return [node[0] for node in self.knowledge_graph.nodes(data=True)]

    def get_all_node_data(self):
        nodes = self.get_all_nodes()
        node_descriptions = dict()
        for node in nodes:
            node_descriptions[node] = list(self.knowledge_graph.edges(node,keys=True,data=True))
        return node_descriptions

    def get_nodes_to_add_to_faiss(self):
        # Return a list of nodes that have been updated, but not added to the FAISS dataset
        return [node[0] for node in self.knowledge_graph.nodes(data=True) if node[1]['updated']>node[1]['added']]

    def get_random_node(self):
        all_nodes = self.get_all_nodes()
        return random.choice(all_nodes)

    def get_neighbors(self,node):
        return [n for n in self.knowledge_graph.neighbors(node)]

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
            
    def check_if_node_exists(self,node):
        return self.knowledge_graph.has_node(node)

    def set_node_summary(self,node,summary):

        if self.knowledge_graph.has_node(node):
            self.knowledge_graph.nodes[node]['summary'] = summary
            self.knowledge_graph.nodes[node]['updated'] = int(time.time())
            return True

        return False

    def get_node_summary(self,node):
        return self.knowledge_graph.nodes[node]['summary'] if self.knowledge_graph.has_node(node) else None

    def get_unparsed_edges(self,node):
        unparsed_edges = [edge for edge in self.knowledge_graph.edges(node,keys=True,data=True) if not(edge[3]['parsed'])]
        return unparsed_edges

    def get_node_summaries(self,nodes):
        return [self.get_node_summary(node) for node in nodes]

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

        alias_pattern = rf"\b({'|'.join(re.escape(alias) for alias in node_aliases)})\b"
        alias_finder = re.compile(alias_pattern, flags=re.IGNORECASE)

        added_edges = []
        for edge in self.knowledge_graph.edges(keys=True,data=True):
            desc = edge[3]['desc']
            src_key = edge[2]

            relevant_nodes = [name for name in list(set(alias_finder.findall(desc))) if len(name)>0]

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
                        print(edge)
                        print("HEre :",src_key)
                        print(f"Added edge '{desc}' between {head} and {tail}")

        for head,tail,desc in added_edges:
            self.add_edge(head,tail,desc)

        print(self.knowledge_graph)
        return added_edges

    def check_nodes_for_replacement(self,named_entities,threshold=0.85):
        nodes = self.get_all_nodes()

        similar_pairs = []
        for key in named_entities:
            similarity = list(map(lambda x :difflib.SequenceMatcher(None, x,key).ratio(),nodes))
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

    def get_edge(self,head,tail):
        # Get all edges between a head and tail node
        return [(head,tail,k,v['desc']) for k,v in self.knowledge_graph.adj[head][tail].items()]

    def parse_edge(self,head,tail,key):
        self.knowledge_graph.edges[head,tail,key]['parsed'] = True

    def get_node_info(self,node):
        # All info is structured as {node_name, node_summary, list((node,tail,info))}

        if(self.knowledge_graph.has_node(node)):
            neighbors = list(self.knowledge_graph.adj[node])

            all_info = dict()

            all_info['node_name'] = node
            all_info['summary'] = self.knowledge_graph.nodes[node]['summary']
            all_info['edge_info'] = []
            for neighbor in neighbors:
                all_info['edge_info'] += self.get_edge(node, neighbor)

            return all_info
        else:
            return None

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

    def save_graph(self,draw_figure=False):

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
        nx.write_gml(self.knowledge_graph, self.knowledge_graph_path)
        self.reload_graph()

    def reload_graph(self):
        self.knowledge_graph = nx.read_gml(self.knowledge_graph_path)

    def load_embedding_model(self,embed_model):
        self.text_embedding_model = embed_model

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
                seed_results.append({'topic':examples['topic'][i],'text':examples['text'][i]})

        for result in seed_results:
            print(f"{result['topic']} : {result['text']}")

        retrieved_context = []
        edge_paths = []
        if(len(seed_results)>1):
            nodes = [s['topic'].strip() for s in seed_results]
            print(nodes)

            # Get paths between all nodes first and resolve before extracting context
            node_pairs = [(nodes[idx],nodes[jdx]) for idx in range(1,len(nodes)) for jdx in range(idx) if not(idx==jdx)]

            # Ignore sub-paths
            all_paths = []
            for source,target in node_pairs:
                if nx.has_path(self.knowledge_graph, source, target):
                    all_paths.append(nx.shortest_path(self.knowledge_graph, source=source, target=target))

            all_paths = sorted(all_paths,key=len,reverse=True)

            filtered_paths = []
            for path in all_paths:
                if(not(any([self.check_subpath(path,c_path) for c_path in filtered_paths]))):
                    filtered_paths.append(path)

            # Get descriptions from filtered paths

            for path in filtered_paths:

                head,tail = path[0],path[-1]
                knowledge_chain = [self.knowledge_graph.nodes[head]['summary']]

                if(len(path)>1):
                    for path_idx in range(len(path)-1):
                        head,tail = path[path_idx:path_idx+2]
                        edge_paths+=[(head,tail,key,edge_dict['desc']) for key,edge_dict in self.knowledge_graph.get_edge_data(head,tail).items()]
                        knowledge_chain += [edge_data['desc'].strip() for edge_data in self.knowledge_graph.get_edge_data(head,tail).values()]

                knowledge_chain += [self.knowledge_graph.nodes[tail]['summary']]
                retrieved_context.append(' '.join(knowledge_chain))    
        else:
            node = seed_results[0]['topic'].strip()
            retrieved_context = [self.knowledge_graph.nodes[node]['summary']]
            retrieved_context += [edge_dict['desc'] for edge_dict in self.knowledge_graph.get_edge_data(node,node).values()]
            edge_paths += [(node,node,key,edge_dict['desc']) for key,edge_dict in self.knowledge_graph.get_edge_data(node,node).items()]
            for neighbor in self.knowledge_graph.neighbors(node):
                retrieved_context += [self.knowledge_graph.nodes[neighbor]['summary']]
                retrieved_context += [edge_dict['desc'] for edge_dict in self.knowledge_graph.get_edge_data(node,neighbor).values()]
                edge_paths += [(node,neighbor,key,edge_dict['desc']) for key,edge_dict in self.knowledge_graph.get_edge_data(node,neighbor).items()]

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

