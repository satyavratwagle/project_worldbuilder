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

class SemanticTools():

    def __init__(self,config):
        self.store_path = config['data_dir']
        self.coreference_model = FCoref(device='cpu')
        self.jsonstore_dir = f'{self.store_path}/json_store'
        self.faiss_dataset_path = f'{self.store_path}/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = f"{self.store_path}/FAISS_store/worldbuilding_dataset.faiss"

    def find_aliases(self,name,corpus):
        #corpus = str

        predictions = self.coreference_model.predict(texts=[corpus])
        clusters = predictions[0].get_clusters()
        all_aliases = []
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

    def load_nli_model(self,nli_model,nli_tokenizer):

        self.nli_tokenizer = nli_tokenizer
        self.nli_model = nli_model

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