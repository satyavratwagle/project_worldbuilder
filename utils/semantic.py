import torch
from gliner import GLiNER
import utils.preprocessing as pre
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from sentence_transformers import SentenceTransformer
import datasets
import networkx as nx
from datasets import Dataset
import faiss
import os
from rank_bm25 import BM25Okapi
import json
from fastcoref import FCoref
import spacy
import psutil
import gc

# Patch to prevent AttributeError with older packages on Transformers 5.x
_orig_getattr = torch.nn.Module.__getattr__
def _patched_getattr(self, name):
    if name == "all_tied_weights_keys":
        return {}
    return _orig_getattr(self, name)
torch.nn.Module.__getattr__ = _patched_getattr
def uppercase(text):
    return text[0].upper()+text[1:]

def print_memory_usage(step):
  process = psutil.Process(os.getpid())
  # rss = Resident Set Size (the actual physical memory used by the process)
  mem_bytes = process.memory_info().rss
  mem_mb = mem_bytes / (1024 * 1024)
  print(f"Step {step} RAM usage: {mem_mb:.2f} MB")

class Semantic():

    def __init__(self):

        #extraction_model_id = "urchade/gliner_small-v2.1"
        #self.extraction_model = GLiNER.from_pretrained(extraction_model_id)

        '''extraction_model_id = "urchade/gliner_small-v2.1"
        nli_model_id        = "cross-encoder/nli-deberta-v3-large"
        embedding_model_id  = "BAAI/bge-m3"

        self.extraction_model = GLiNER.from_pretrained(extraction_model_id)

        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model_id)
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(nli_model_id)

        self.text_embedding_model = SentenceTransformer(embedding_model_id)

        STORE_DIR = "data/FAISS_store/"
        self.dataset = datasets.load_from_disk(os.path.join(STORE_DIR, "obsidian_dataset_temp"))
        self.dataset.load_faiss_index("embeddings", os.path.join(STORE_DIR, "obsidian_index_temp.faiss"))'''

        with open('config.json', "r", encoding="utf-8") as f:
            # Load the JSON data into a Python dictionary
            config = json.load(f)
        self.store_path = config['data_dir']

        self.coreference_model = FCoref(device='cpu')

    def find_aliases(self,name,corpus):
        #corpus = str

        predictions = self.coreference_model.predict(texts=[corpus])
        clusters = predictions[0].get_clusters()
        for cluster in clusters:
            if(name in cluster):
                all_aliases = list(set(cluster))
                break

        # remove pronouns
        # Load the lightweight spaCy model
        nlp = spacy.load("en_core_web_sm")

        # Remove apostrophes and pronouns
        aliases = []
    
        for phrase in all_aliases:
            doc = nlp(phrase)
            # Filter out tokens tagged as possessive ('POS')
            # Alternatively, use token.lemma_ or just drop the POS token and keep the rest
            filtered_tokens = [token.text for token in doc if token.pos_ != "PRON" and token.tag_ != "POS"]
            cleaned_phrase = " ".join(filtered_tokens).strip()
            if cleaned_phrase:
                aliases.append(cleaned_phrase)

        aliases = list(set(aliases))
                
        return aliases

    def load_extraction_model(self,extraction_model):
        self.extraction_model = extraction_model

    def load_embedding_model(self,embed_model,dataset):
        self.text_embedding_model = embed_model
        self.dataset = dataset

    def load_nli_model(self,nli_model,nli_tokenizer):

        self.nli_tokenizer = nli_tokenizer
        self.nli_model = nli_model

    def create_new_dataset(self,documents):

        new_dataset = Dataset.from_list(documents)

        def embed_text(batch):
            # Return a dictionary mapping to your target index string key
            print_memory_usage('Batching')
            return {"embeddings": self.text_embedding_model.encode(batch["text"], normalize_embeddings=True)}
            
        embedding_dim = self.text_embedding_model.get_embedding_dimension()
        cosine_index = faiss.IndexFlatIP(embedding_dim)
        self.new_dataset = new_dataset.map(embed_text, batched=True, batch_size=8)
        torch.mps.empty_cache()

        return self.new_dataset,cosine_index

    def reload_faiss_dataset(self):

        self.dataset = datasets.load_from_disk(os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset"))
        self.dataset.load_faiss_index("embeddings", os.path.join(self.store_path, "FAISS_store/worldbuilding_dataset.faiss"))
        print_memory_usage('Before')
        try:
            del self.new_dataset
        except:
            pass
        print_memory_usage('After')

        print(self.dataset)

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

    def check_context_entailment(self,contexts,queries):

        for context in contexts:
            print(context)
            for query in queries:
                c,e,n, = self.get_entailment_probs(context,query)
                print(f"{query} : {np.round(c,2)} | {np.round(e,2)} | {np.round(n,2)}")

    def entity_extraction(self,text,labels=['character','location','artifact','faction','event'],entity_threshold=0.45):

        entities = self.extraction_model.predict_entities(text, labels)

        chosen_entities = []
        for e in entities:
            if(e['score']>entity_threshold):
                chosen_entities.append(e['text'])

        entities_dict = dict()
        for e in entities:
            if(e['text'] in chosen_entities):
                entities_dict[e['text']] = e['label']

        return entities_dict

    def named_entity_extraction(self,corpus ,labels=['character','location','artifact','faction','event'],entity_threshold=0.7):

        #corpus = pre.get_corpus(corpus_path)
        chunks = pre.get_chunks(corpus,chunk_size=200)

        entities_dict = dict()
        for chunk in chunks:

            entities = self.extraction_model.predict_entities(chunk, labels)
            for ent in entities:

                if(not(ent['text'] in entities_dict.keys())):
                    entities_dict[ent['text']]= {label:[] for label in labels}

                entities_dict[ent['text']][ent['label']].append(ent['score'])

            print(entities)
            print(chunk)
            print()


        chosen_entities = dict()
        for key in entities_dict.keys():
            if(key[0].isupper()):

                scores = []
                for label in labels:
                    score = np.nan_to_num(np.asarray(np.mean(entities_dict[key][label])),0.0)
                    scores.append(score)

                if(np.max(scores)>entity_threshold):
                    chosen_entities[key] = labels[np.argmax(scores)]

        return chosen_entities

    def get_entailment_probs(self,premise,hypothesis):
        #premise = f"Context: {context}\nPremise: {premise}"
        #hypothesis = f"hypothesis: {hypothesis}"
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = self.nli_model(**inputs).logits
        # Label index 1 is typically 'entailment' in cross-encoder models
        probs = torch.softmax(logits, dim=-1).squeeze()
        return probs[0],probs[1],probs[2] # Contradiction, Entailment, Neutral

    def check_entailment(self,premise: str, hypothesis: str, context:str) -> bool:

        #premise = f"Context: {context}\nPremise: {premise}"
        #hypothesis = f"hypothesis: {hypothesis}"
        inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = nli_model(**inputs).logits
        # Label index 1 is typically 'entailment' in cross-encoder models
        probs = torch.softmax(logits, dim=1).squeeze()
        pred_label = torch.argmax(probs).item()
        return pred_label == 1  # Returns True if premise entails hypothesis

    def is_semantically_equivalent(self,s1: str, s2: str, context:str) -> bool:
        return check_entailment(s1, s2, context) and check_entailment(s2, s1, context)

    def compute_semantic_entropy_and_consistency(self,samples, context, sample_probs=None):
        N = len(samples)
        if sample_probs is None:
            sample_probs = np.ones(N) / N  # Black-box uniform weighting
            
        clusters = []  # List of lists containing sample indices
        
        # 2. Greedy Semantic Clustering
        for i, sample in enumerate(samples):
            assigned = False
            for cluster in clusters:
                rep_sample = samples[cluster[0]]
                if is_semantically_equivalent(sample, rep_sample, context):
                    cluster.append(i)
                    assigned = True
                    break
            if not assigned:
                clusters.append([i])
                
        # 3. Aggregate Probabilities
        cluster_probs = np.array([sum(sample_probs[i] for i in cluster) for cluster in clusters])
        normalized_cluster_probs = cluster_probs / np.sum(cluster_probs)
        
        # 4. Calculate Semantic Entropy
        # Adding 1e-12 inside log to prevent log(0)
        semantic_entropy = -np.sum(normalized_cluster_probs * np.log(normalized_cluster_probs + 1e-12))
        
        # 5. Calculate Consistency Score (Dominant Cluster Size / N)
        max_cluster_size = max(len(cluster) for cluster in clusters)
        consistency_score = max_cluster_size / N
        
        return {
            "semantic_entropy": float(semantic_entropy),
            "consistency_score": float(consistency_score),
            "num_clusters": len(clusters)
        }