import torch
from gliner import GLiNER
import numpy as np
import datasets
import os
from fastcoref import FCoref
import spacy
import psutil
from fastcoref import spacy_component
from gliclass import GLiClassModel, ZeroShotClassificationPipeline


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

        self.config = config
        self.store_path = config['data_dir']
        self.coreference_model = FCoref(device='cpu')
        self.jsonstore_dir = f'{self.store_path}/json_store'
        self.faiss_dataset_path = f'{self.store_path}/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = f"{self.store_path}/FAISS_store/worldbuilding_dataset.faiss"

        self.nlp = spacy.load("en_core_web_lg")
        self.fcoref = FCoref(device="mps")
        #self.nlp.add_pipe("fastcoref")
        #self.classifier = pipeline('zero-shot-classification',model='cross-encoder/nli-deberta-v3-large')

    def find_aliases(self,doc,entity_coreferences,entity_dict):

        preds = self.fcoref.predict(doc.text)

        for cluster,entity_name in entity_coreferences:

            aliases = []
            for start,end in cluster:
                span = doc.char_span(start,end)

                if(any([(token.pos_=='PROPN' and not token.dep_=='poss') for token in span])):
                    entity_dict[entity_name]['aliases'].append(span.text)

        for key in entity_dict.keys():
            entity_dict[key]['aliases'] = list(set(entity_dict[key]['aliases']))

        return entity_dict

    def get_root_entity(self,doc,start,end):
        # Get the root entity within a span of text defined by start and end

        for ent in doc.ents:
            if not (end <= ent.start_char or start >= ent.end_char):
                if(ent.label_ in ["PERSON","ORG","WORK_OF_ART"]):
                    print(doc.text[start:end],ent.label_)
                    # Clean entity name
                    cleaned_entity = []
                    for token in ent:
                        if(token.pos_ == "PART" and token.text.lower() == "'s"):
                            continue
                        if(token.text.lower() != token.lemma_.lower()):
                            word = token.lemma_
                            if token.pos_ == "PROPN":
                                word = word.capitalize()
                        else:
                            word = token.text
                        cleaned_entity.append(word + token.whitespace_)

                    return "".join(cleaned_entity).strip()

    def extract_pos(self,text,pos=None):

        #nlp = spacy.load("en_core_web_lg")
        doc = self.nlp(text)

        if(pos):
            extracted_pos_tokens = [token for token in doc if any(token.pos_ == p for p in pos)]
        else:
            extracted_pos_tokens = [token for token in doc]

        return extracted_pos_tokens,doc

    def split_sentence(self,text):

        doc = self.nlp(text)

        for token in doc:
            if(token.dep_=='ROOT'):
                split_at = token.text
                break

        print(text,split_at)
        head,tail = text.split(split_at)

        return head.strip(),tail.strip()

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

    def resolve_coreferences(self,doc,entity_names):

        clusters = doc._.coref_clusters
        all_replacements = []
        for cluster in clusters:
            for selected_cluster,entity,aiases in entity_names:
                if(set(cluster) == set(selected_cluster)):
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

        final_text = doc.text
        for start, end, replacement in filtered_replacements:
            final_text = final_text[:start] + replacement + final_text[end:]

        return final_text

    def get_alias_clusters(self, text, entity_dict):

        self.nlp.add_pipe("fastcoref")

        doc = self.nlp(text,component_cfg={"fastcoref": {"resolve_text": True}})

        # Access the fully resolved text via spaCy's custom extension
        resolved_output = doc._.resolved_text
        clusters = doc._.coref_clusters

        print("Entities : ",[(ent.text, ent.label_) for ent in doc.ents])
        print("Resolved : ",resolved_output)

        entity_coreferences = []

        for cluster in clusters:
            entities_in_cluster = []
            for start,end in cluster:

                print('\nDoc Cluster : ',doc.text[start:end])
                root_entity = self.get_root_entity(doc,start,end)
                print(root_entity)
                if(root_entity):
                    entities_in_cluster.append(root_entity)

            # Set primary alias for cluster

            set_primary_alias = False
            entities_in_cluster = set(entities_in_cluster)
            for selection in entity_dict.keys():
                if(any([alias in entities_in_cluster for alias in entity_dict[selection]['aliases']])):
                    primary_alias = selection
                    entities_in_cluster.update(set(entity_dict[selection]['aliases']))
                    set_primary_alias = True

            if(not(set_primary_alias)):
                if(len(entities_in_cluster)>0):
                    primary_alias = list(entities_in_cluster)[0]
                else:
                    primary_alias = doc.text[cluster[0][0]:cluster[0][1]]

            entity_coreferences.append((cluster,primary_alias,list(entities_in_cluster)))
            
        self.nlp.remove_pipe("fastcoref")

        return doc,entity_coreferences

    # Load Models
    def load_zsc_model(self):
        # Load Zero-Shot Classification Model
        self.zsc_model = GLiClassModel.from_pretrained(self.config['zero_shot_classification_model'])
        self.zsc_tokenizer = AutoTokenizer.from_pretrained(self.config['zero_shot_classification_model'])
        self.zsc_model.config.prompt_first = True
        self.zsc_model.config.normalize_features = True
        self.zsc_model.config.pooling_strategy = "first"

        self.zsc_pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')
        self.zsc_pipeline.pipe.sep_token = self.zsc_tokenizer.sep_token

    def load_extraction_model(self):
        self.extraction_model = GLiNER.from_pretrained(self.config['ner_extraction_model'])

    def load_nli_model(self):

        nli_max_length = 512

        # Load and update the configuration to accommodate larger token lengths
        config = AutoConfig.from_pretrained(self.config['nli_model'])
        config.max_position_embeddings = nli_max_length
        config.max_relative_positions = nli_max_length
        
        self.nli_tokenizer = AutoTokenizer.from_pretrained(self.config['nli_model'])
        self.nli_tokenizer.model_max_length = nli_max_length
        self.nli_model = AutoModelForSequenceClassification.from_pretrained(self.config['nli_model'],config=config)

    # Semantic Functions

    def relationship_extraction(self, texts, topics=[]):

        # texts : list(str)

        # Make object properties later if needed.
        relation_labels = [
                    "location; place; position; coordinates",
                    "cause",
                    "property; quality; status; tag",
                    "time; duration; era; time period"
                ]

        entity_labels = [topic+['object'] for topic in topics]

        _, relations = self.extraction_model.inference(
        texts=texts,
        labels=entity_labels,
        relations=relation_labels,
        threshold=0.6,
        adjacency_threshold=0.3,
        relation_threshold=0.3,
        return_relations=True,
        multi_label = True,
        flat_ner=False
            )

        triplets = []
        for sample in relations:
            sample_triplets = []
            for r in sample:
                sample_triplets.append((r['head']['text'],r['tail']['text'],r['relation'],r['score']))
                print(f"{r['head']['text']} --> {r['tail']['text']} : {r['relation']} ({r['score']})")
            triplets.append(sample_triplets)

        # Filter Triplets
        filtered_triplets = []
        for idx,sample_triplets in enumerate(triplets):

            best_score = 0.0
            best_score_with_topic = 0.0
            best_triplet_with_topic = None
            for sample_triplet in sample_triplets:

                # Keep the triplet with a topic in the head and highest score
                if((sample_triplet[3]>best_score_with_topic) and (topics[idx][0] in sample_triplet[0])):
                    best_score_with_topic = sample_triplet[3]
                    best_triplet_with_topic = sample_triplet

                # Keep the triplet with the highest absolute score.
                if(sample_triplet[3]>best_score):
                    best_score = sample_triplet[3]
                    best_triplet = sample_triplet


            # Choose the triplet with topic. If none, go with the topic with the highest score.
            if(best_triplet_with_topic):
                filtered_triplets.append(best_triplet_with_topic)
            else:
                filtered_triplets.append(best_triplet)

            print(texts[idx])
            print(filtered_triplets[-1])
            print()



        return None

    def zero_shot_classification(self,text,labels,prompt=None,threshold=0.8,examples=[]):
        # Important! Create a pipeline before this step
        # Multi-class Zero-shot classification to generate tags
        # texts (str) : Text to classify
        # labels (list)     : Labels to classify as
        # threshold (float) : Threshold beyond which a label is considered True

        # returns labels (list(list([label,score]))) : Tags for each string in texts.

        '''pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')

        if(prompt):
            results = pipeline(texts, labels, prompt=prompt, threshold=threshold)
        else:
            results = pipeline(texts, labels, threshold=threshold)

        return [[(score_tuple['label'],score_tuple['score']) for score_tuple in result] for result in results]
        '''
        assert self.zsc_pipeline

        results = results = self.zsc_pipeline(text,labels,prompt=prompt,threshold=threshold,examples=examples)

        return results

    def edge_labelling(self,edge_text,topic,threshold=0.5,rac_examples=[]):

        prompt = f"Classify this description of a {topic}."

        pos_tokens,doc = self.extract_pos(edge_text)

        print([(token,token.dep_,token.pos_) for token in doc])

        head_verb = ''
        aux_verb = ''
        for token in doc:
            if(token.pos_=='AUX'):
                aux_verb = token.text
            if(token.pos_=='VERB' and token.dep_=='ROOT'):
                head_verb = token.text
            if(token.text==topic):
                print(token,token.dep_,token.head,token.head.pos_,list(token.head.rights))
                #head_verb = token.head


        labels = [f"Where {aux_verb} {topic} {head_verb}?",
                f"Why{aux_verb} {topic}{head_verb}?",
                f"How{aux_verb} {topic}{head_verb}?",
                f"What{aux_verb} {topic}{head_verb}?",
                f"How long {aux_verb} {topic} {head_verb}?"]

        labels = [f"{topic}{aux_verb}{head_verb} in this place.",
                f"{topic}{aux_verb}{head_verb} because of this.",
                f"{topic}{aux_verb}{head_verb} via this process.",
                f"{topic} has these qualities, adjectives or attributes.",
                f"{topic} was true for this duration of time."]

        if(len(head_verb)==0):
            head_verb = aux_verb

        labels = [f"{head_verb} : Geographical location, place, spatial description, relative position",
                f"{head_verb} : Procedure, methodology, cause, effect",
                f"{head_verb} : Attribute, quality, adjective, property, unique descriptor",
                f"{head_verb} : Time, relative time, temporal description, era, duration"]

        #results = self.zero_shot_classification(edge_text,labels,threshold=threshold,prompt=prompt,examples=rac_examples)
        results = self.zero_shot_classification(edge_text,labels,threshold=threshold,prompt=prompt,examples=rac_examples)

        edge_text = edge_text.replace("Hatyaars","Hatyaars (faction)")
        entities,relations = self.extraction_model.inference(texts=edge_text,labels=[head_verb,'verb, action word',topic,'other'],relations=labels,threshold=0.3,relation_threshold=0.1,return_relations=True,flat_ner=False,multi_label=True)
        for e in entities[0]:
            print(e)

        print()
        for r in relations[0]:
            print(f"{r['head']['text']} --> {r['tail']['text']} : {r['relation']:<105} ({r['score']})")

        return results

    def legacy_zero_shot_classification(self,texts,labels,context=None,threshold=0.8,prompt=None):
        # Multi-class Zero-shot classification to generate tags
        # texts (list(str)) : Text to classify
        # labels (list)     : Labels to classify as
        # threshold (float) : Threshold beyond which a label is considered True

        # returns labels (list(list([label,score]))) : Tags for each string in texts.

        '''pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')

        if(prompt):
            results = pipeline(texts, labels, prompt=prompt, threshold=threshold)
        else:
            results = pipeline(texts, labels, threshold=threshold)

        return [[(score_tuple['label'],score_tuple['score']) for score_tuple in result] for result in results]
        '''

        if(context):
            texts = texts+self.zsc_tokenizer.sep_token+"".join(context)

        pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')
        pipeline.pipe.sep_token = self.zsc_tokenizer.sep_token
        #res = pipeline([texts],labels,threshold=0.0)


        tokenized_inputs = pipeline.pipe.prepare_inputs([texts],labels,True,examples=None,prompt=None)
        input_ids = tokenized_inputs["input_ids"][0]
        sep_indices = (input_ids == self.zsc_tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()

        outputs = self.zsc_model.model.encoder_model(
                                        tokenized_inputs["input_ids"],
                                        attention_mask=tokenized_inputs["attention_mask"],
                                        output_attentions=True,
                                        output_hidden_states=True,
                                        return_dict=False
                                    )
        final_hidden_states = outputs[0]
        #print(final_hidden_states.shape)

        logits, loss, pooled_output, classes_embedding = self.zsc_model.model.process_encoder_output(tokenized_inputs["input_ids"],
                                                                                                        tokenized_inputs['attention_mask'],
                                                                                                        final_hidden_states,max_num_classes=len(labels))

        _, _, text_token_embeddings, text_mask = self.zsc_model.model._extract_class_features(final_hidden_states, tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], len(labels))

        filtered_hidden_states = text_token_embeddings#[:,sep_indices[0]:sep_indices[1],:]
        manual_pooled_output = filtered_hidden_states
        manual_pooled_output = self.zsc_model.model.pooler(filtered_hidden_states)
        manual_pooled_output = self.zsc_model.model.text_projector(manual_pooled_output)
        manual_pooled_output = self.zsc_model.model.dropout(manual_pooled_output)

        scores = torch.sigmoid(torch.einsum("BD,BCD->BC", manual_pooled_output, classes_embedding)).detach().numpy()[0]

        results = [[(labels[i],scores[i]) for i in range(len(labels))]]

        return results

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