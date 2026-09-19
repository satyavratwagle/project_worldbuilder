import re
import json
import copy

with open('utils/prompts.json', "r", encoding="utf-8") as f:
    prompts = json.load(f)

def process_prompt(template_dict,args_dict=None):

	prompt_dict = copy.deepcopy(template_dict)

	if(prompt_dict['role']=='system'):
		prompt_dict['content'] = "\n".join(prompt_dict['content'])

	else:
		prompt_text = ""
		if(args_dict):
			for key in prompt_dict['content'].keys():
				current_prompt_block = "".join(prompt_dict['content'][key])
				block_modified = False

				for block_key,value in args_dict.items():
					if(len(value)>0 and (f'({block_key.upper()})' in current_prompt_block)):
						current_prompt_block = current_prompt_block.replace(f'({block_key.upper()})',args_dict[block_key])
						block_modified = True

				if(block_modified):
					prompt_text += current_prompt_block

		prompt_dict['content'] = prompt_text

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

def get_node_wiki_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'node_name' in args_dict.keys()
	assert 'node_description' in args_dict.keys()
	assert 'node_property' in args_dict.keys()

	history.append(process_prompt(prompts['node_wiki_prompt']['system']))
	history.append(process_prompt(prompts['node_wiki_prompt']['user'],args_dict))
	print(history[-1]['content'])
	return history

def get_node_summary_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'node_name' in args_dict.keys()
	assert 'node_description' in args_dict.keys()
	assert 'existing_summary' in args_dict.keys()

	if(len(args_dict['existing_summary'])>0):
		history.append(process_prompt(prompts['node_summary_prompt']['system']))
		history.append(process_prompt(prompts['node_summary_prompt']['user'],args_dict))

	else:
		history.append(process_prompt(prompts['node_summary_prompt']['system']))
		history.append(process_prompt(prompts['node_summary_prompt']['user'],args_dict))
	return history

def get_text_decomposition_prompt(args_dict,history=[]):
	# topics, text
	assert 'topics' in args_dict.keys()
	assert 'text' in args_dict.keys()
	assert 'background_knowledge' in args_dict.keys()

	history.append(process_prompt(prompts['text_decomposition_prompt']['system']))
	history.append(process_prompt(prompts['text_decomposition_prompt']['user'],args_dict))
	print(history[0]['content'])

	return history

def get_reasoned_generation_prompt(args_dict,history=[]):

	assert 'user_query' in args_dict.keys()
	assert 'local_context' in args_dict.keys()

	if(len(history)==0):
		history.append(process_prompt(prompts['reasoned_answer_prompt']['system']))
	history.append(process_prompt(prompts['reasoned_answer_prompt']['user'],args_dict))

	print(history[-1]['content'])

	return history

def get_brainstorming_prompt(args_dict,history=[]):

	assert 'topic' in args_dict.keys()
	assert 'local_context' in args_dict.keys()

	if(len(history)==0):
		history.append(process_prompt(prompts['brainstorming_prompt']['system']))
	history.append(process_prompt(prompts['brainstorming_prompt']['user'],args_dict))
	print(history[-1]['content'])

	return history

def get_hypothesizing_prompt(args_dict,history=[]):

	assert 'topic' in args_dict.keys()
	assert 'facts' in args_dict.keys()

	if(len(history)==0):
		history.append(process_prompt(prompts['hypothesizing_prompt']['system']))
	history.append(process_prompt(prompts['hypothesizing_prompt']['user'],args_dict))
	print(history[-1]['content'])

	return history

def get_chain_of_thought_regex():
	reasoning_structure_regex = (
    r"<scratchpad>\n[\s\S]*?\n</scratchpad>\n+"
    r"Final Answer:\s*[\s\S]+"
	)

	return reasoning_structure_regex