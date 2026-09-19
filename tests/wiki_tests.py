from utils.semantic import SemanticTools
from utils.datastore_utils import DatastoreUtilities
from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer
import json
import random
import copy
import torch
import numpy as np
from prettytable import PrettyTable

import matplotlib.pyplot as plt
import shap

import torch.nn.functional as F
class WikiTests:

    def __init__(self):

        with open('config.json', "r", encoding="utf-8") as f:
            # Load the JSON data into a Python dictionary
            config = json.load(f)

        self.du = DatastoreUtilities(config)
        self.sem = SemanticTools(config)

        # This model is a uni-encoder
        self.sem.load_zsc_model("knowledgator/gliclass-modern-base-v3.0")
        self.zsc_model = self.sem.zsc_model
        self.zsc_tokenizer = self.sem.zsc_tokenizer
        self.zsc_model.config.prompt_first = True
        self.zsc_model.config.pooling_strategy = "avg"
        print(self.zsc_model.config.architecture_type)
        print(self.zsc_model.config.normalize_features)
        print(self.zsc_model.config.prompt_first)
        print(self.zsc_model.config.encoder_layer_id)
        print(self.zsc_model.config.scorer_type)
        #print("Special Tokens Map:", self.zsc_tokenizer.special_tokens_map)

    def test_edge_label(self):

        for test_edge in self.du.knowledge_graph.edges("marushar",keys=True,data=True):


            edge_text = test_edge[3]['desc']

            endpoints = list(set(test_edge[3]['endpoints']))
            endpoint_types = [self.du.knowledge_graph.nodes[key]['type'] for key in test_edge[3]['tags'].keys()]
            endpoint_summaries = [self.du.knowledge_graph.nodes[key]['summary'] for key in test_edge[3]['tags'].keys()]

            with open('utils/prompts.json', "r", encoding="utf-8") as f:
                prompts = json.load(f)

            zsc_prompt = prompts['edge_classification_prompt']

            with open('utils/wiki_schema.json', "r", encoding="utf-8") as f:
                wiki_schema = json.load(f)

            def process_wiki_schema(src_dict,topic=None):

                wiki_dict = copy.deepcopy(src_dict)
                
                if(topic):
                    for key,value in wiki_dict.items():
                        wiki_dict[key] = wiki_dict[key].replace(f'[TOPIC]',topic)

                return wiki_dict

            for idx in range(len(endpoints)):

                topic       = endpoints[idx]
                topic_type  = endpoint_types[idx]

                schema = process_wiki_schema(wiki_schema[topic_type.lower()],topic)
                prompt = zsc_prompt.replace("[TOPIC]",topic)
                prompt = prompt.replace("[SCHEMA]","\n".join(key+" : "+schema[key] for key in schema.keys()))

                # Marushar lies south of the Shovzogr ocean.[SEP]Marushar is a desert continent located south of the Shovzogr ocean.Shovzogr is a dead ocean located south of the Archipelago with still water.
                # Marushar lies south of the Shovzogr ocean.[SEP]Marushar is a desert continent located south of the Shovzogr ocean.Shovzogr is a dead ocean located south of the Archipelago with still water.
                #labels = [key.lower() for key in schema.keys()]
                labels = [f"{schema[key]}" for key in schema.keys()]
                table = PrettyTable(['Masked']+[label for label in labels])

                zsc_input = edge_text+self.zsc_tokenizer.sep_token+"".join(endpoint_summaries)
                print(zsc_input)
                pipeline = ZeroShotClassificationPipeline(self.zsc_model, self.zsc_tokenizer, classification_type='multi-label', device='mps')
                pipeline.pipe.sep_token = self.zsc_tokenizer.sep_token
                results = pipeline([zsc_input], labels, threshold=0.0)
                print(f"{zsc_input} :",[r['score'] for r in results[0]])

                
                print()

                # MANUAL POOLING

                tokenized_inputs = pipeline.pipe.prepare_inputs([zsc_input],labels,True,examples=None,prompt=None)
                input_ids = tokenized_inputs["input_ids"][0]
                sep_indices = (input_ids == self.zsc_tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()
                #print(sep_indices)
                #print("Labels  : ",self.zsc_tokenizer.decode(input_ids[:sep_indices[0]]))
                ##print("Text    : ",self.zsc_tokenizer.decode(input_ids[sep_indices[0]+1:sep_indices[1]]))
                #print("Context : ",self.zsc_tokenizer.decode(input_ids[sep_indices[1]+1:sep_indices[2]]))

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
                #filtered_hidden_states = torch.concat([final_hidden_states[:,:3,:],final_hidden_states[:,11:,:]],dim=1)
                #filtered_hidden_states = torch.mean(text_token_embeddings,dim=1)

                '''
                # Check semantic similarity between labels
                #print(classes_embedding.shape)
                normalized_labels = F.normalize(classes_embedding[0], p=2, dim=1) # Using first token anchor as proxy
                # Compute pairwise cosine similarity matrix
                similarity_matrix = torch.matmul(normalized_labels, normalized_labels.T).detach().numpy()

                schema_keys = list(schema.keys())
                table = PrettyTable([' ']+[label for label in schema.keys()])
                for idx in range(len(similarity_matrix)):
                    table.add_row([schema_keys[idx]]+[f"{s:.3f}" for s in similarity_matrix[idx]])

                #print(table)
                '''


                #print(text_token_embeddings.shape)
                filtered_hidden_states = text_token_embeddings[:,sep_indices[0]:sep_indices[1],:]
                #print(filtered_hidden_states.shape)
                #print(classes_embedding.shape)
                manual_pooled_output = filtered_hidden_states
                manual_pooled_output = self.zsc_model.model.pooler(filtered_hidden_states)
                manual_pooled_output = self.zsc_model.model.text_projector(manual_pooled_output)
                manual_pooled_output = self.zsc_model.model.dropout(manual_pooled_output)

                logits = torch.einsum("BD,BCD->BC", manual_pooled_output, classes_embedding) #self.zsc_model.model.scorer(manual_pooled_output,classes_embedding)
                print("Text : ",[f"{val:.3f}" for val in torch.sigmoid(logits).detach().numpy()[0]])

                filtered_hidden_states = text_token_embeddings[:,sep_indices[1]:,:]
                #print(filtered_hidden_states.shape)
                #print(classes_embedding.shape)
                manual_pooled_output = filtered_hidden_states
                manual_pooled_output = self.zsc_model.model.pooler(filtered_hidden_states)
                manual_pooled_output = self.zsc_model.model.text_projector(manual_pooled_output)
                manual_pooled_output = self.zsc_model.model.dropout(manual_pooled_output)

                logits = torch.einsum("BD,BCD->BC", manual_pooled_output, classes_embedding) #self.zsc_model.model.scorer(manual_pooled_output,classes_embedding)
                print("Context : ",[f"{val:.3f}" for val in torch.sigmoid(logits).detach().numpy()[0]])

                filtered_hidden_states = text_token_embeddings[:,sep_indices[0]:,:]
                #print(filtered_hidden_states.shape)
                #print(classes_embedding.shape)
                manual_pooled_output = filtered_hidden_states
                manual_pooled_output = self.zsc_model.model.pooler(filtered_hidden_states)
                manual_pooled_output = self.zsc_model.model.text_projector(manual_pooled_output)
                manual_pooled_output = self.zsc_model.model.dropout(manual_pooled_output)

                logits = torch.einsum("BD,BCD->BC", manual_pooled_output, classes_embedding) #self.zsc_model.model.scorer(manual_pooled_output,classes_embedding)
                print("Text + Context : ",[f"{val:.3f}" for val in torch.sigmoid(logits).detach().numpy()[0]])
                print(labels)



            #print("\n-------\n")

            # SHAP Values
            '''
            target_label = 'person'
            labels = ['person','location','object','event']

            def f(texts):
                print(texts)
                results = pipeline(texts,labels,threshold=0.0)

                return_values = []
                for res in results:
                    for s in res:
                        print(s)
                        if(s['label']==target_label):
                            return_values.append(s['score'])
                print(return_values)

                return return_values

            explainer = shap.Explainer(f, shap.maskers.Text(r"W+",mask_token=" "))
            shap_results = explainer([zsc_input])
            
            for idx in range(len(shap_results.values[0])):
                print(shap_results.data[0][idx]," : ",shap_results.values[0][idx])
            '''

            #tokenized_inputs = pipeline.pipe.prepare_inputs([zsc_input],labels,True)
            #input_text = []
            #nput_text.append(pipeline.pipe.prepare_input(zsc_input,labels))
            #print(input_text)

            '''
            inputs = []
            inputs.append(pipeline.pipe.prepare_input(zsc_input, labels))
            tokenized_inputs = pipeline.pipe.tokenizer(inputs, return_offsets_mapping=True, max_length=len(labels), padding="longest", return_tensors="pt")

            input_ids = tokenized_inputs["input_ids"]
            print(len(input_ids))
            offset_mapping = tokenized_inputs["offset_mapping"][0] # Character (start, end) for each token

            # Extract tokens and their positions
            print(f"{'Index':<6} | {'Token ID':<10} | {'Token String':<15} | {'Char Span':<10}")
            print("-" * 50)

            token_strs = []
            for kdx in range(len(input_ids)):
                for idx,token_id in enumerate(input_ids[kdx]):
                    token_str = self.zsc_tokenizer.decode([token_id])
                    token_strs.append(token_str)
                    
                    print(f"{idx:<6} | {token_id.item():<10} | {token_str:<15}")

            

            # THE FOLLOWING CODE WORKS.
            results = pipeline.pipe.get_embeddings([zsc_input],labels)
            logits = self.zsc_model.model.scorer(torch.from_numpy(results[0]['text_embedding']).unsqueeze(0),torch.from_numpy(results[0]['class_embeddings']).unsqueeze(0))
            print('Using pipe.get_embeddings : ',torch.sigmoid(logits))



            zsc_input = pipeline.pipe._normalize_texts([zsc_input])
            labels = pipeline.pipe._process_labels(labels)
            tokenized_inputs = pipeline.pipe.prepare_inputs(zsc_input,labels,True,examples=None,prompt=None)
            input_ids = tokenized_inputs["input_ids"][0]

            sep_indices = (input_ids == self.zsc_tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()
            print(sep_indices)

            outputs = pipeline.pipe.model(**tokenized_inputs,
                                    labels=None,
                                    max_num_classes=len(labels),
                                    output_text_embeddings=True,
                                    output_class_embeddings=True,
                                    output_hidden_states=True,
                                    output_attentions=True)

            
            logits = self.zsc_model.model.scorer(outputs.text_embeddings,outputs.class_embeddings)
            print('Using pipe.model : ',torch.sigmoid(logits))
            # Works up to here

            outputs = self.zsc_model.model.encoder_model(
                                            tokenized_inputs["input_ids"],
                                            attention_mask=tokenized_inputs["attention_mask"],
                                            output_attentions=True,
                                            output_hidden_states=True,
                                            return_dict=False
                                        )
            final_hidden_states = outputs[0]
            logits, loss, pooled_output, classes_embedding = self.zsc_model.model.process_encoder_output(tokenized_inputs["input_ids"],
                                                                                                            tokenized_inputs['attention_mask'],
                                                                                                            final_hidden_states,max_num_classes=len(labels))
            print("Using model.process_encoder_output : ",torch.sigmoid(logits))

            # Manually pool vectors

            _, _, text_token_embeddings, text_mask = self.zsc_model.model._extract_class_features(final_hidden_states, tokenized_inputs["input_ids"], tokenized_inputs["attention_mask"], len(labels))
            #filtered_hidden_states = torch.concat([final_hidden_states[:,:3,:],final_hidden_states[:,11:,:]],dim=1)
            #filtered_hidden_states = torch.mean(text_token_embeddings,dim=1)

            for i in range(sep_indices[0],sep_indices[1]):
                filtered_hidden_states = text_token_embeddings[:,i,:]
                #print(filtered_hidden_states.shape)
                #print(classes_embedding.shape)
                manual_pooled_output = filtered_hidden_states
                #manual_pooled_output = self.zsc_model.model.pooler(filtered_hidden_states)
                manual_pooled_output = self.zsc_model.model.text_projector(manual_pooled_output)
                manual_pooled_output = self.zsc_model.model.dropout(manual_pooled_output)

                logits = torch.einsum("BD,BCD->BC", manual_pooled_output, classes_embedding) #self.zsc_model.model.scorer(manual_pooled_output,classes_embedding)
                print("Manual pooling : ",token_strs[i],' : ',torch.sigmoid(logits))
            print(labels)

            #print(pooled_output.shape,classes_embedding.shape)

            #sep_indices = (input_ids == self.zsc_tokenizer.sep_token_id).nonzero(as_tuple=True)[0].tolist()
            '''

            '''print(tokens)
            print("Num tokens : ",len(tokens))
            print("Final hidden shape : ",final_hidden_states.shape)
            print(sep_indices)
            if sep_indices:
                start_target_idx = sep_indices[0] + 1 
                label_end_idx = sep_indices[0]
                # The second separator typically marks the boundary before the label tokens section
                end_target_idx = sep_indices[1] if len(sep_indices) > 1 else len(input_ids)
            else:
                start_target_idx = 0
                end_target_idx = len(input_ids)

            # CHANGE THIS TO SLICE VECTORS!
            print(start_target_idx,end_target_idx)
            print(tokens[start_target_idx:end_target_idx])
            target_pooled_vector = final_hidden_states[0, start_target_idx:end_target_idx].mean(dim=0)
            print(target_pooled_vector.shape)
            label_start_idx = sep_indices[-1] + 1 if len(sep_indices) > 1 else end_target_idx
            print(label_start_idx)
            print(tokens[1:label_end_idx])
            label_pooled_vector = final_hidden_states[0, 1:label_end_idx].mean(dim=0)
            print(label_pooled_vector.shape)

            manual_logits = []
            raw_logits = outputs.logits[0]

            label_embeddings_slice = final_hidden_states[0,1:label_end_idx]'''
            
            #for i in range(5):

        
            #torch.sigmoid(torch.einsum("BD,BCD->BC", pooled_output, classes_embedding))
            #manual_logit = torch.dot(pooled_output[0], classes_embedding[0][i])
            #print(labels[i],torch.sigmoid(manual_logit).item())
            
            

            #print(manual_scores)
            
            # Attention Rollout
            '''
            attentions = outputs.attentions
            num_layers = len(attentions)
            batch_size, num_heads, seq_len, _ = attentions[0].shape

            # Compute Attention Rollout
            rollout_matrix = torch.eye(seq_len)
            for l in range(num_layers):
              layer_attn = attentions[l].squeeze(0).mean(dim=0)
              identity = torch.eye(seq_len)
              augmented_attn = 0.5 * layer_attn + 0.5 * identity
              augmented_attn = augmented_attn / augmented_attn.sum(dim=-1, keepdim=True)
              rollout_matrix = torch.matmul(rollout_matrix, augmented_attn)

            # --- FINDING SEMANTIC MAGNETS ---
            # Sum across columns (dim=0) to see total incoming attention received by each token
            incoming_attention = rollout_matrix
            # Row tells you the attention given by a token
            # Column tells you the attention pulled by a token

            tokens = []
            for idx in range(len(tokenized_inputs["input_ids"][0])):
                tokens.append(self.zsc_tokenizer.decode(tokenized_inputs["input_ids"][0][idx]))
            print(tokens)


            for idx,token in enumerate(tokens):
                for jdx,token2 in enumerate(tokens):
                    if(token2==' location'):
                        print(tokens[idx],f" : {(rollout_matrix[jdx,idx]).detach().numpy():.3f}")
            '''

            '''
            values = [f"{v:.2f}" for k,v in results[0]]
            table.add_row(["Unmasked"]+values)

            for idx in range(len(tokens)):

                # Apply custom attention mask
                custom_attention_mask = torch.ones_like(tokenized_inputs['attention_mask'])
                custom_attention_mask[0][idx] = 0
                # Run a forward pass
                tokenized_inputs['attention_mask'] = custom_attention_mask

                model_output = self.zsc_model(**tokenized_inputs,max_num_classes=len(labels),output_text_embeddings=True,output_class_embeddings=True)
                #results = pipeline.get_embeddings([zsc_input],labels)
                scores = torch.sigmoid(model_output.logits.detach()).numpy()
                #

                values = [f"{v:.2f}" for v in scores[0]]
                table.add_row([tokens[idx]]+values)
                
            print(table)
            '''



wiki_test = WikiTests()
wiki_test.test_edge_label()