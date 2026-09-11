import re
import json
import copy

with open('utils/prompts.json', "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    prompts = json.load(f)

def process_prompt(template_dict,args_dict=None):

	prompt_dict = copy.deepcopy(template_dict)
	prompt_dict['content'] = ''.join(prompt_dict['content'])

	if(args_dict):
		for key,value in args_dict.items():
			prompt_dict['content'] = prompt_dict['content'].replace(f'[{key.upper()}]',args_dict[key])

	return prompt_dict

# See if we need a class later.
class PromptParser:

	def __init__(self,required_keys):

		# required_keys : str (Keys that are used by the prompt)
		self.required_keys = required_keys

	def __str__(self):
		return ",".join(self.required_keys)

	def assert_keys(self,args_dict):
		assert all([required_key in args_dict.keys() for required_key in self.required_keys])

def get_node_summary_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'node_name' in args_dict.keys()
	assert 'node_description' in args_dict.keys()
	assert 'existing_summary' in args_dict.keys()

	if(len(args_dict['existing_summary'])>0):
		history.append(process_prompt(prompts['node_summary_prompt_existing']['system']))
		history.append(process_prompt(prompts['node_summary_prompt_existing']['user'],args_dict))

	else:
		history.append(process_prompt(prompts['node_summary_prompt_new']['system']))
		history.append(process_prompt(prompts['node_summary_prompt_new']['user'],args_dict))
	return history

# was simple_summary
def get_text_decomposition_prompt(args_dict,history=[]):
	# topics, text
	assert 'topics' in args_dict.keys()
	assert 'text' in args_dict.keys()

	history.append(process_prompt(prompts['text_decomposition_prompt']['system']))
	history.append(process_prompt(prompts['text_decomposition_prompt']['user'],args_dict))

	return history

# was generation_prompt
def get_reasoned_generation_prompt(args_dict,history=[]):

	assert 'user_query' in args_dict.keys()
	assert 'local_context' in args_dict.keys()

	history.append(process_prompt(prompts['reasoned_answer_prompt']['system']))
	history.append(process_prompt(prompts['reasoned_answer_prompt']['user'],args_dict))

	return history

def get_chain_of_thought_regex():
	reasoning_structure_regex = (
    r"<scratchpad>\n[\s\S]*?\n</scratchpad>\n+"
    r"Final Answer:\s*[\s\S]+"
	)

	return reasoning_structure_regex