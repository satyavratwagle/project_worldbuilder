import json
import utils.prompts as prompts
#from gliclass import ZeroShotClassificationPipeline
import ollama
from pydantic import BaseModel,Field
from typing import Literal, Optional
from utils.pydantic_schema import QueryEvaluationSchema
import utils.prompts as prompts

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

with open('utils/tools.json', "r", encoding="utf-8") as f:
    tools = json.load(f)

tools_list = [tool_desc for tool_desc in tools.values()]

def load_markdown(file_name,start_para, end_para):

    with open(f"{config['data_dir']}/obsidian_data/{file_name}.md", "r", encoding="utf-8") as f:
        md_text = f.readlines()

    curr_para = 0
    text_to_return = ''
    for row in md_text:
        if(row.startswith('- ')):
            if(curr_para>end_para):
                return text_to_return
            elif(curr_para>=start_para):
                print(curr_para, text_to_return)
                text_to_return += row[1:].strip()+' '

            curr_para += 1

    return text_to_return


context = load_markdown('ambar',0,1)
query = "What is Ambar and how does it affect Mahamun?"

props_dict = dict()
props_dict['context'] = context
props_dict['query'] = query


prompt = prompts.get_query_evaluation_prompt(props_dict,[])

print(prompt)

response = ollama.chat(
                                model='llama3.1:8b',
                                messages=prompt,
                                #tools=tools_json,
                                options={
                                            'temperature': 0.1,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                            'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
                                        },
                                format=QueryEvaluationSchema.model_json_schema()
                                )

print(response.eval_duration)
print(prompt[-1]['content'])
llm_response = QueryEvaluationSchema.model_validate_json(response.message.content)

print("\n\n---\n\n")
if(llm_response.is_context_sufficient):
    print(f"{llm_response.genre} : Answer --> {llm_response.answer}")
else:
    print(f"Context not sufficient!\n{llm_response.genre} : Explanation --> {llm_response.new_query}")

    props_dict = dict()
    props_dict['query'] = llm_response.new_query

    prompt = prompts.get_tool_use_prompt(props_dict,[])

    response = ollama.chat(
                                    model='llama3.1:8b',
                                    messages=prompt,
                                    tools=tools_list,
                                    options={
                                                'temperature': 0.1,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                                                'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
                                            },
                                    )

    print(response.message.tool_calls)
