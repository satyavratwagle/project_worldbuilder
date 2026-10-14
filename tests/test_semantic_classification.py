import json
from utils.datastore_utils import DatastoreUtilities
from transformers import pipeline
import ollama
from utils.pydantic_schema import NodeSummary, RelationshipCategory
import utils.prompts as prompts

with open('utils/semantic_definitions.json', "r", encoding="utf-8") as f:
    definitions = json.load(f)

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='worldbuilding')
#du.load_embedding_model()

print([du.get_node_name(node) for node in du.knowledge_graph.nodes()])

all_nodes = [node for node in du.knowledge_graph.nodes() if du.get_node_layer(node)=='topic']
for head_node in all_nodes:
	for tail_node in all_nodes:

		if(not(head_node==tail_node)):
		
			head_neighbors = set([node for node in du.knowledge_graph.neighbors(head_node) if du.get_node_layer(node)=='lore'])
			tail_neighbors = set([node for node in du.knowledge_graph.neighbors(tail_node) if du.get_node_layer(node)=='lore'])

			common_neighbors = list(head_neighbors & tail_neighbors)

			

			if(len(common_neighbors)>0):

				args_dict = dict()
				args_dict['head'] = f"{du.get_node_name(head_node)} : {du.get_node_summary(head_node)}"
				args_dict['tail'] = f"{du.get_node_name(tail_node)} : {du.get_node_summary(tail_node)}"
				args_dict['relationship'] = " ".join([du.get_node_name(node) for node in common_neighbors])

				history = prompts.get_relationship_categorization_prompt(args_dict,[])

				response = ollama.chat(
		                        model='llama3.1:8b',
		                        messages=history,
		                        options={
		                                    'temperature': 0.3,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
		                                    'num_predict': 32       # Equivalent to max tokens (maximum tokens to generate)
		                                },
		                        format=RelationshipCategory.model_json_schema()
		                        )

				response = RelationshipCategory.model_validate_json(response.message.content)
				
				if(not(response.category=='Other')):
					print(history[-1]['content'])
					print(f"{du.get_node_name(head_node)} ---> {du.get_node_name(tail_node)}")
					print(response.category)
					print()