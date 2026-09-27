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
query = "Where does Mahamun live and how does he get his powers?"

props_dict = dict()
props_dict['query'] = query

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

print(response)