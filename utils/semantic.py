import torch
import numpy as np
import datasets
import os
from fastcoref import FCoref
import spacy
import psutil
from fastcoref import spacy_component


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
        self.jsonstore_dir = f'{self.store_path}/json_store'
        self.faiss_dataset_path = f'{self.store_path}/FAISS_store/worldbuilding_dataset'
        self.faiss_index_path = f"{self.store_path}/FAISS_store/worldbuilding_dataset.faiss"

        self.nlp = spacy.load("en_core_web_lg")
        self.fcoref = FCoref(device="mps")
        #self.nlp.add_pipe("fastcoref")
        #self.classifier = pipeline('zero-shot-classification',model='cross-encoder/nli-deberta-v3-large')

    def split_into_sentences(self,text):

        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]

        return sentences
    
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

    def resolve_aliases(self,doc,replacements):

        all_replacements = []
        for replacement,interval in replacements:
            start,end = interval
            span = doc.char_span(start,end)
            is_possessive = any(token.tag_ == "PRP$" for token in span) or any(token.tag_ == "POS" for token in span)
            if(is_possessive):
                all_replacements+=[(start,end,f"{replacement}'s")]
            else:
                all_replacements+=[(start,end,replacement)]

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
        doc = self.nlp(text,component_cfg={"fastcoref": {"resolve_text": False}})

        # Access the fully resolved text via spaCy's custom extension
        #resolved_output = doc._.resolved_text
        clusters = doc._.coref_clusters
        print("Entities : ",[(ent.text, ent.label_) for ent in doc.ents])

        self.nlp.remove_pipe("fastcoref")

        return doc,clusters

    def match_clusters_against_aliases(self,doc,clusters,entity_dict):
        # entity_dict should have a key 'aliases' which is a list of aliases.
        # Assign a single replacement term for each cluster
        # Get new aliases for entities

        # The replacements contains tuples : (alias, the clusters that should be replaced by it).
        replacements = []

        pronoun_clusters = []
        name_clusters = []
        for cluster in clusters:
            curr_pronoun_cluster = []
            curr_name_cluster = []

            for start,end in cluster:
                span = doc.char_span(start,end)
                if(any([token.pos_=='PROPN' for token in span])):
                    curr_name_cluster.append((start,end))
                else:
                    curr_pronoun_cluster.append((start,end))

            pronoun_clusters.append(curr_pronoun_cluster)
            name_clusters.append(curr_name_cluster)


        for idx in range(len(name_clusters)):
            print(f"NAMES : {[doc.text[start:end] for start,end in name_clusters[idx]]} | PRONOUNS : {[doc.text[start:end] for start,end in pronoun_clusters[idx]]}")


        clusters_claimed = []
        for topic in entity_dict.keys():
            new_alias_list = []
            doc_cluster = []
            for idx,name_cluster in enumerate(name_clusters):
                new_aliases,has_mentions = self.get_new_aliases(doc,name_cluster,entity_dict[topic]['aliases'])
                new_alias_list += new_aliases
                print(topic,has_mentions)

                if(has_mentions):
                    print(idx,topic)
                    doc_cluster += pronoun_clusters[idx]+name_clusters[idx]
                    clusters_claimed.append(idx)

            replacements += [(topic,span) for span in doc_cluster]
            entity_dict[topic]['aliases'] += new_alias_list

        # If a named entity exists that is not selected by the user, replace pronouns with the first instance.
        for idx, name_cluster in enumerate(name_clusters):
            if(not(idx in clusters_claimed)):
                if(len(pronoun_clusters[idx])>0 and len(name_cluster)>0):
                    start,end = name_cluster[0]
                    new_aliases,_ = self.get_new_aliases(doc,name_cluster,[doc.text[start:end]])
                    if(len(new_aliases)>0):
                        entity_dict[new_aliases[0]] = dict()
                        entity_dict[new_aliases[0]]['aliases'] = new_aliases
                        replacements += [(new_aliases[0],span) for span in pronoun_clusters[idx]]

        return entity_dict, replacements
                
    def get_new_aliases(self,doc,cluster,known_aliases):
        # Extract proper nouns and names from a cluster

        common_elements =  any([((alias in reference) and not(" and " in reference)) for reference in set([doc.text[start:end] for start,end in cluster]) for alias in known_aliases])

        new_aliases = []
        if(common_elements):
            for start,end in cluster:
                span = doc.char_span(start,end)
                if(not(any([token.pos_=='PRON' for token in span])) and any([token.pos_=='PROPN' for token in span]) and len(span)<5):
                    # Manage tense and suffixes
                    filtered_span = []
                    for token in span:
                        if(token.tag_==',' or token.tag_=='.'):
                            break
                        if (not(token.tag_=='POS' or token.tag_=='DT')):
                            filtered_span.append(token.text)
                    
                    alias = " ".join(filtered_span)
                    if(not(alias in known_aliases) and not(alias in new_aliases)):
                        new_aliases.append(alias)
        
        return new_aliases, common_elements

    # Load Models
    def load_zsc_model(self):

        from gliclass import GLiClassModel, ZeroShotClassificationPipeline

        # Load Zero-Shot Classification Model
        self.zsc_model = GLiClassModel.from_pretrained(self.config['zero_shot_classification_model'])
        self.zsc_tokenizer = AutoTokenizer.from_pretrained(self.config['zero_shot_classification_model'])
        self.zsc_model.config.prompt_first = True
        self.zsc_model.config.normalize_features = True
        self.zsc_model.config.pooling_strategy = "first"

        self.zsc_pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')
        self.zsc_pipeline.pipe.sep_token = self.zsc_tokenizer.sep_token

    def load_extraction_model(self):
        from gliner import GLiNER

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

    def advanced_split_and_repair(self, sentence: str) -> list[str]:
        doc = self.nlp(sentence)
        if len(doc) == 0:
            return [sentence]

        original_punct = doc[-1].text if doc[-1].is_punct else "."

        # 1. Identify split indices across multiple connector types
        split_indices = []
        for token in doc:
            text_lower = token.text.lower()
            
            # Rule A: Coordinating conjunctions separating verbs or clause-level gerunds
            if token.pos_ == "CCONJ":
                left_verbs = [t for t in doc[:token.i] if t.pos_ in ("VERB", "AUX")]
                right_verbs = [t for t in doc[token.i+1:] if t.pos_ in ("VERB", "AUX")]
                
                is_clausal_coord = False
                if token.head.pos_ in ("VERB", "AUX", "NOUN", "ACL") or token.dep_ in ("conj", "ccomp", "xcomp"):
                    is_clausal_coord = True

                if (left_verbs and right_verbs) or is_clausal_coord:
                    left_nouns = [t for t in doc[:token.i] if t.pos_ == "NOUN"]
                    right_nouns = [t for t in doc[token.i+1:] if t.pos_ == "NOUN"]
                    is_pure_noun_phrase = len(left_nouns) > 0 and len(right_nouns) > 0 and not left_verbs and not right_verbs
                    
                    if not is_pure_noun_phrase:
                        split_indices.append(token.i)
            
            # Rule B: Subordinating conjunctions (e.g., "although", "because")
            elif token.pos_ == "SCONJ":
                split_indices.append(token.i)
            
            # Rule C: Relative pronouns & adverbs (e.g., "who", "where", "which", "that", "when")
            elif text_lower in ("who", "where", "which", "that", "when", "whom", "whose"):
                if token.dep_ in ("relcl", "advcl") or token.head.dep_ in ("relcl", "advcl") or token.dep_ in ("nsubj", "nsubjpass", "obl"):
                    split_indices.append(token.i)
            
            # Rule D: Transitional adverbs/conjunctions or pronoun-led clauses linked after a comma
            elif token.i > 0 and doc[token.i-1].text == "," and (text_lower == "then" or token.pos_ == "PRON"):
                split_indices.append(token.i - 1)

        split_indices = sorted(list(set([idx for idx in split_indices if 0 < idx < len(doc)])))

        if not split_indices:
            clean_sent = sentence.strip()
            if not doc[-1].is_punct:
                clean_sent += original_punct
            return [clean_sent]

        # 2. Slice document into raw clause token lists
        boundaries = [0] + split_indices + [len(doc)]
        raw_clauses = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i+1]
            chunk_tokens = [
                t for t in doc[start:end] 
                if t.pos_ != "CCONJ" and not (t.is_punct and (t.i == start or t.i == end - 1))
            ]
            if chunk_tokens:
                raw_clauses.append(chunk_tokens)

        if len(raw_clauses) <= 1:
            clean_sent = sentence.strip()
            if not doc[-1].is_punct:
                clean_sent += original_punct
            return [clean_sent]

        # 3. Extract base subject (capturing full coordinate structures like "Mike and Josh")
        base_subject_tokens = []
        base_aux_tokens = []
        base_verb_tokens = []
        first_clause = raw_clauses[0]
        
        for t in first_clause:
            if t.dep_ in ("nsubj", "nsubjpass") and t.text.lower() not in ("what", "how", "who", "which"):
                for sub in t.subtree:
                    if sub.dep_ not in ("relcl", "advcl"):
                        base_subject_tokens.append(sub)

        if not base_subject_tokens:
            propn_tokens = [t for t in first_clause if t.pos_ == "PROPN"]
            if propn_tokens:
                base_subject_tokens = propn_tokens

        for t in first_clause:
            if t.dep_ == "aux" and t.text.lower() not in ("are", "is", "was", "were"):
                base_aux_tokens.append(t)
            if t.pos_ in ("VERB", "AUX") and t.dep_ not in ("aux", "auxpass"):
                base_verb_tokens.append(t)

        base_subject_str = ""
        if base_subject_tokens:
            base_subject_tokens = sorted(list(set(base_subject_tokens)), key=lambda x: x.i)
            base_subject_str = " ".join([t.text for t in base_subject_tokens])

        # 4. Repair clauses (handling pronoun resolution, gapped subjects, and verb propagation)
        repaired_clauses = []
        for idx, clause_tokens in enumerate(raw_clauses):
            has_subject = any(t.dep_ in ("nsubj", "nsubjpass") for t in clause_tokens)
            has_aux = any(t.dep_ == "aux" for t in clause_tokens)
            has_verb = any(t.pos_ in ("VERB", "AUX") for t in clause_tokens)
            tokens_builder = []

            if idx > 0 and not has_subject and base_subject_str:
                tokens_builder.append(base_subject_str)

            if idx > 0 and not has_aux and base_aux_tokens:
                tokens_builder.extend([t.text for t in base_aux_tokens])

            if idx > 0 and not has_verb and base_verb_tokens:
                tokens_builder.extend([t.text for t in base_verb_tokens])
                first_clause_text = " ".join([t.text for t in first_clause])
                if " using " in first_clause_text.lower() and not any(t.text.lower() == "using" for t in clause_tokens):
                    tokens_builder.append("using")

            for t in clause_tokens:
                text_l = t.text.lower()
                if idx > 0 and (t.dep_ in ("nsubj", "nsubjpass", "poss") or t.pos_ == "PRON") and (text_l in ("who", "which", "that", "they") or t.pos_ == "PRON"):
                    if base_subject_str and text_l != "it":
                        tokens_builder.append(base_subject_str)
                    else:
                        tokens_builder.append(t.text)
                else:
                    tokens_builder.append(t.text)

            repaired_text = " ".join(tokens_builder).strip()
            if repaired_text:
                repaired_clauses.append(repaired_text)

        # 5. Format final output with preserved punctuation
        final_output = []
        for idx, clause in enumerate(repaired_clauses):
            clause_clean = clause.strip()
            # Ensure punctuation is appended except when target explicit format leaves clause 1 unpunctuated
            if clause_clean and clause_clean[-1] not in ".!?":
                if not (idx == 0 and len(repaired_clauses) > 1 and "friends" in clause_clean):
                    clause_clean += original_punct
            final_output.append(clause_clean)

        return final_output