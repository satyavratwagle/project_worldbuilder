import json
from gliner import GLiNER
from utils.semantic import SemanticTools
import utils.prompts as prompts
#from gliclass import ZeroShotClassificationPipeline
import ollama
from pydantic import BaseModel,Field
from typing import Literal

class TextTuple(BaseModel):
    atomic_sentence: str = Field(description="An atomic sentence.")
    topic: str = Field(description="Subject of 'atomic_sentence'")
    label: Literal["location", "process", "attribute", "time period"] = Field(description="Label assigned to 'atomic_sentence'")

class DecomposedText(BaseModel):
	decomposed_text: list[TextTuple]


with open('config.json', "r", encoding="utf-8") as f:
	config = json.load(f)


def load_markdown(file_name,para_number):

	with open(f"{config['data_dir']}/obsidian_data/{file_name}.md", "r", encoding="utf-8") as f:
		md_text = f.readlines()

	curr_para = 0
	for row in md_text:
		if(row.startswith('- ')):
			if(curr_para==para_number):
				return row[1:].strip()
			else:
				curr_para += 1

graph_name = "graph_json_20260922_162006.json"
with open(f'{config['data_dir']}/{graph_name}', "r", encoding="utf-8") as f:
	graph = json.load(f)

nodes = graph['nodes']
edges = graph['edges']

#sem = SemanticTools(config)
#sem.load_zsc_model()
#sem.load_extraction_model()

ref_text = load_markdown('timekeeping',0)
topics = ['Crossing','Forday','Prahar']
topic_type = 'event'

props_dict = dict()
props_dict['topics']                = '\n'.join([f"- {t}" for idx,t in enumerate(topics)])
props_dict['text']                  = ref_text

chat_history = prompts.get_text_decomposition_prompt(props_dict,history=[])

print(chat_history[-1]['content'])

response = ollama.chat(
                                model='llama3.1:8b',
                                messages=chat_history,
                                #tools=tools_json,
                                options={
                                            'temperature': 0.2,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                            'num_predict': 1024       # Equivalent to max tokens (maximum tokens to generate)
                                        },
                                format=DecomposedText.model_json_schema()
                                )

print(ref_text)
print("\n\n---\n\n")
print(response.message.content)

llm_response = DecomposedText.model_validate_json(response.message.content)

print("\n\n---\n\n")
print('Refined : ')
for tuplet in llm_response.decomposed_text:
	print(f"{tuplet.topic} --> {tuplet.atomic_sentence} : {tuplet.label}")

'''
for refined_edge_text in llm_response.decomposed_text:
	results = sem.edge_labelling(refined_edge_text.strip()+'.',topic,threshold=0.0,rac_examples=[])
	print(refined_edge_text)
	for r in results[0]:
		print(f"{r['label']:<65} ({r['score']})")
	print()
'''

'''
for edge in edges:

	#entity_labels = list(set([nodes[edge['head']]['name'],nodes[edge['tail']]['name']]))+['other']

	#extracted_pos,doc = sem.extract_pos(edge['text'])
	#head,relation,tail = sem.get_triplets(extracted_pos,doc)

	#entity_labels = list(set([nodes[edge['head']]['name'],nodes[edge['tail']]['name']]))+['other']

	#sem.zero_shot_classification(edge['text'],labels=relation_labels)

	if(nodes[edge['head']]['name']=='Mahamun'):

		refined_edge_text = edge['text']#.replace(nodes[edge['head']]['name'],f"{nodes[edge['head']]['name']} ({nodes[edge['head']]['type']})")
		results = sem.edge_labelling(refined_edge_text,nodes[edge['head']]['name'],threshold=0.0)

		print(refined_edge_text)
		for r in results[0]:
			print(f"{r['label']:<65} ({r['score']})")
		print()
'''