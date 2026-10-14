import ollama
import utils.prompts as prompts
from utils.pydantic_schema import SufficiencyCheck
from utils.datastore_utils import DatastoreUtilities
from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

# is a discrete, tangible, non-sentient material object, artifact or substance that can be possessed, utilized, or consumed.

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='worldbuilding')
du.load_embedding_model()
sem = SemanticTools(config)

text = "What climate does Marushar have?"

clauses = sem.advanced_split_and_repair(text)
context, nodes, edges = du.get_graph_rag_context(user_query=clauses,strategy='explore_neighborhood')

args_dict = dict()
args_dict['context'] = context
args_dict['query'] = text

history = prompts.get_context_sufficiency_check_prompt(args_dict,[])

print(history[-1]['content'])

response = ollama.chat(
                        model='llama3.1:8b',
                        messages=history,
                        options={
                                    'temperature': 0.3,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                    'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
                                },
                        format=SufficiencyCheck.model_json_schema()
                        )

response = SufficiencyCheck.model_validate_json(response.message.content)

print(response.is_sufficient)
print(response.relevant_information)
print(response.information_needed)
print(response.additional_questions)

