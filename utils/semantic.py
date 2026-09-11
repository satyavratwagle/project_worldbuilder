import torch
from gliner import GLiNER
import utils.preprocessing as pre
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer,pipeline
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
from fastcoref import spacy_component
from collections import deque


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

        self.nlp = spacy.load("en_core_web_lg", exclude=["parser", "lemmatizer", "ner", "textcat"])
        self.nlp.add_pipe("fastcoref")
        #self.classifier = pipeline('zero-shot-classification',model='cross-encoder/nli-deberta-v3-large')

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
        nlp = spacy.load("en_core_web_lg")

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

    def extract_pos(self,text,pos=None):

        nlp = spacy.load("en_core_web_lg")
        doc = nlp(text)

        if(pos):
            extracted_pos_tokens = [token for token in doc if any(token.pos_ == p for p in pos)]
        else:
            extracted_pos_tokens = [token for token in doc]

        return extracted_pos_tokens,doc

    def get_triplets(self,token_list,doc=None):

        pos_list = [t.pos_ for t in token_list]
        text_list = [t.text for t in token_list]


        relation = None
        head = None
        tail = None
        remove_pos = [text_list[idx] if pos_list[idx]=='DET' else None for idx in range(len(text_list))]

        if('VERB' in pos_list):
            remove_pos += [text_list[idx] if pos_list[idx]=='AUX' else None for idx in range(len(text_list))]
            verb_idx = pos_list.index('VERB')
            if(verb_idx<(len(pos_list)-1)):
                if(pos_list[verb_idx+1]=='ADP'):
                    relation = text_list[verb_idx]+' '+text_list[verb_idx+1]
                    head = doc[:token_list[verb_idx].i]
                    if((verb_idx+1)<(len(pos_list)-1)):
                        tail = doc[token_list[verb_idx+2].i:]
                    else:
                        tail = doc[token_list[verb_idx+1].i:]
                else:
                    relation = text_list[verb_idx]
                    head = doc[:token_list[verb_idx].i]
                    tail = doc[token_list[verb_idx+1].i:]
                    
            else:
                relation = text_list[verb_idx]
                head = doc[:token_list[verb_idx].i]
                tail = doc[token_list[verb_idx].i:]
        else:
            if('AUX' in pos_list):
                aux_idx = pos_list.index('AUX')
                if(aux_idx<(len(pos_list)-1)):
                    if(pos_list[aux_idx+1]=='ADP'):
                        relation = text_list[aux_idx]+' '+text_list[aux_idx+1]
                        head = doc[:token_list[aux_idx].i]
                        if((aux_idx+1)<(len(pos_list)-1)):
                            tail = doc[token_list[aux_idx+2].i:]
                        else:
                            tail = doc[token_list[aux_idx+1].i:]

                    else:
                        relation = text_list[aux_idx]
                        head = doc[:token_list[aux_idx].i]
                        tail = doc[token_list[aux_idx+1].i:]
                else:
                    relation = text_list[aux_idx]
                    head = doc[:token_list[aux_idx].i]
                    tail = doc[token_list[aux_idx].i:]
                
        if(relation):
            head = head.text
            tail = tail.text
            for r in remove_pos:
                if(r):
                    head.replace(r+" ","")
                    tail.replace(r+" ","")
            

        return head,relation,tail

    def resolve_coreferences(self,resolved_text,entity_names):

        #resolved_text,_ = self.nlp(text,component_cfg={"fastcoref": {"resolve_text": True}})
        doc = self.nlp(resolved_text,component_cfg={"fastcoref": {"resolve_text": False}})
        clusters = doc._.coref_clusters
        cluster_text = [[resolved_text[c[0]:c[1]] for c in cluster] for cluster in clusters]

        all_replacements = []
        for cluster in doc._.coref_clusters:
            cluster_aliases = [resolved_text[start:end] for start,end in cluster]
            for selected_cluster,entity in entity_names:
                print(list(set(selected_cluster)))
                print(list(set(cluster_aliases)))
                print()
                if(set(selected_cluster).issubset(set(cluster_aliases))):
                    for start,end in cluster:
                        span = doc.char_span(start,end)
                        is_possessive = any(token.tag_ == "PRP$" for token in span) or any(token.tag_ == "POS" for token in span)
                        if(is_possessive):
                            all_replacements+=[(start,end,f"{entity}'s")]
                        else:
                            all_replacements+=[(start,end,entity)]

        # Sort from right to left to prevent index shifting
        all_replacements.sort(key=lambda x: x[0], reverse=True)

        # Remove overlapping replacements
        filtered_replacements = []
        min_start_seen = float('inf')
        
        for start, end, replacement in all_replacements:
            if end <= min_start_seen:
                filtered_replacements.append((start, end, replacement))
                min_start_seen = start
            else:
                continue

        final_text = resolved_text
        for start, end, replacement in filtered_replacements:
            final_text = final_text[:start] + replacement + final_text[end:]

        return final_text

    def get_coref_clusters(self, text, new_text=None):

        nlp = spacy.load("en_core_web_lg", exclude=["parser", "lemmatizer", "ner", "textcat"])
        nlp.add_pipe("fastcoref")

        if(new_text):
            doc = nlp(f"{text}\n\n{new_text}",component_cfg={"fastcoref": {"resolve_text": True}})
        else:
            doc = nlp(text,component_cfg={"fastcoref": {"resolve_text": True}})

        # 5. Access the fully resolved text via spaCy's custom extension
        resolved_output = doc._.resolved_text
        clusters = doc._.coref_clusters

        cluster_text = [list(set([text[c[0]:c[1]] for c in cluster])) for cluster in clusters]

        '''resolved_output = ''
        for token in doc:
            resolved_output += (token.text + token.whitespace_)'''

        #print("--- Original Text ---")
        #print(text)

        #print("\n--- Resolved Text ---")
        #print(resolved_output)

        return resolved_output,cluster_text,doc

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

    def entity_extraction_chunk(self,chunk,labels=['character','location','artifact','faction','event','time'],entity_threshold=0.7):

        chunk = '.\n'.join([c.strip() for c in chunk.split('.')])
        entities = self.extraction_model.inference(
                texts=[chunk],
                labels=labels,
                threshold=entity_threshold,
                return_relations=False,
                flat_ner=True
            )

        return entities

    def named_entity_extraction(self,corpus ,labels=['character','location','artifact','faction','event','time'],entity_threshold=0.7):

        chunks = corpus.split('\n\n')
        extracted_entities = dict()
        for chunk in chunks:
            entities = self.entity_extraction_chunk(chunk,labels,entity_threshold)
            for e in entities[0]:
                extracted_entities[e['text']] = e['label']

        return extracted_entities

    def get_entailment_probs(self,premise,hypothesis,context=None):
        #premise = f"Context: {context}\nPremise: {premise}"
        #hypothesis = f"hypothesis: {hypothesis}"

        #label_mapping = ['likely','unlikely']
        #res = self.classifier(premise+hypothesis, label_mapping)
        #print(res)

        if(context):
            premise = f"[CONTEXT] {context} [PREMISE] {premise}"
        inputs = self.nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
        print(f"Cumulative token length is : {inputs["input_ids"].shape[1]}")
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