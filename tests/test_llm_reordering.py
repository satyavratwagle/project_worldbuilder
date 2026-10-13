import ollama
import utils.prompts as prompts
from utils.pydantic_schema import PhaseList
from utils.datastore_utils import DatastoreUtilities
from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='test',current_graph='lore')
du.load_embedding_model()
sem = SemanticTools(config)

text = "Before Eva Stratt had recruited Ryland Grace, the academic circles had disowned Ryland Grace."

clauses = clauses = sem.advanced_split_and_repair(text)
print(f"\advanced_split_and_repair : {text} ---> {clauses}")
#context, lore_nodes = du.get_graph_rag_context(clauses,strategy='find_path_between',add_code=True,context_label='phases')
nodes = du.find_nodes_in_clauses(clauses,threshold=0.7,filter='plot')


ordered_events = []
while(len(nodes)>0):
    context, node_dict = du.get_plot_to_reorder(nodes)
    print(node_dict)
        
    args_dict = dict()

    args_dict['phases'] = context
    args_dict['outline'] = text

    history = prompts.get_reordering_prompt(args_dict,[])

    #print(history[-1]['content'])

    response = ollama.chat(
                            model='llama3.1:8b',
                            messages=history,
                            options={
                                        'temperature': 0.0,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                        'num_predict': 64       # Equivalent to max tokens (maximum tokens to generate)
                                    },
                            format=PhaseList.model_json_schema()
                            )

    first_event = PhaseList.model_validate_json(response.message.content)

    event_id = first_event.event_id

    print(first_event.thinking)
    print(event_id)

    print()

    
    if(not(event_id in node_dict.keys())):
        break
    else:
        ordered_events.append(event_id)
        node_dict.pop(event_id)
        nodes = list(node_dict.values())
