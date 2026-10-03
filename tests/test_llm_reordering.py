from utils.datastore_utils import DatastoreUtilities
import os
import json
from pathlib import Path
import re
import ollama
import utils.prompts as prompts
from utils.pydantic_schema import PhaseList

og_text = "Ryland Grace is a scientist who has been kicked out of academia because his peers don't believe him. He is later sent into space to save the world from Astrophage by Eva Stratt against his will. Later, he meets an alien named Rocky in outer space."

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

du = DatastoreUtilities(config,project_name='demo',current_graph='lore',is_core_graph=True)


check_phases = True
order_text = False


if(check_phases):

    existing_phases = du.format_plot_edges()
    unordered_edges = du.get_unordered_plot_edges()

    print(existing_phases)

    for edge in unordered_edges:
        print(f"Unordered : {edge[3]}")

if(order_text):

    du.load_embedding_model()

    text_to_reorder, reference_dict = du.get_plot_to_reorder(og_text)
    print(text_to_reorder)

    args_dict = dict()

    args_dict['phases'] = du.format_plot_edges()
    args_dict['outline'] = og_text
    args_dict['text'] = text_to_reorder

    history = prompts.get_reordering_prompt(args_dict,[])

    print(history[-1]['content'])

    response = ollama.chat(
                            model='llama3.1:8b',
                            messages=history,
                            options={
                                        'temperature': 0.0,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                        'num_predict': 1024       # Equivalent to max tokens (maximum tokens to generate)
                                    },
                            format=PhaseList.model_json_schema()
                            )

    ordered_phases = PhaseList.model_validate_json(response.message.content)

    for phase in ordered_phases.phases:
        current_phase = int(phase.index)
        for id in phase.ids:
            head,tail,key,text = reference_dict[id]
            du.set_plot_edge_phase(head,tail,key,current_phase)
            print(f"Assigned '{text}' to phase {current_phase}")
        print()

    du.save_graph()

