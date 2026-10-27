import re
import json
import copy
import chainlit as cl

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

def refresh_history():

	history = cl.user_session.get("chat_history")
	history = []
	cl.user_session.set("chat_history",history)
	
def prompt_parser(prompt_name,args_dict,refresh=True):

	if(refresh):
		refresh_history()

	prompt_template = prompts[prompt_name]
	required_keys = prompt_template['required_keys']

	assert all([required_key in args_dict.keys() for required_key in required_keys])

	history = cl.user_session.get("chat_history")

	if(len(history)==0):
		history.append(process_prompt(prompts[prompt_name]['system']))
	else:
		history[0] = process_prompt(prompts[prompt_name]['system'])
	history.append(process_prompt(prompts[prompt_name]['user'],args_dict))

	cl.user_session.set("chat_history",history)
	return history

def add_to_history(role,content):

	history = cl.user_session.get("chat_history")
	history.append({"role":role,"content":content})
	cl.user_session.set("chat_history",history)


def switch_system_prompt(prompt_name):

	history = cl.user_session.get("chat_history")
	if(len(history)==0):
		history.append(process_prompt(prompts[prompt_name]['system']))
	else:
		history[0] = process_prompt(prompts[prompt_name]['system'])
	cl.user_session.set("chat_history",history)

'''
def get_tool_use_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'query' in args_dict.keys()

	history.append(process_prompt(prompts['tool_use_prompt']['system']))
	history.append(process_prompt(prompts['tool_use_prompt']['user'],args_dict))
	return history

def get_node_definition_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'node_name' in args_dict.keys()
	assert 'node_description' in args_dict.keys()

	history.append(process_prompt(prompts['node_definition_prompt']['system']))
	history.append(process_prompt(prompts['node_definition_prompt']['user'],args_dict))
	print(history[-1]['content'])
	return history

def get_node_definition_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'node_name' in args_dict.keys()
	assert 'information' in args_dict.keys()
	
	history.append(process_prompt(prompts['node_definition_prompt']['system']))
	history.append(process_prompt(prompts['node_definition_prompt']['user'],args_dict))
	return history

def get_entity_discovery_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'statements' in args_dict.keys()
	assert 'known_topics' in args_dict.keys()
	
	history.append(process_prompt(prompts['entity_discovery_prompt']['system']))
	history.append(process_prompt(prompts['entity_discovery_prompt']['user'],args_dict))
	return history

def get_relationship_categorization_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'statement' in args_dict.keys()
	
	history.append(process_prompt(prompts['relationship_categorization_prompt']['system']))
	history.append(process_prompt(prompts['relationship_categorization_prompt']['user'],args_dict))
	return history

def get_entity_categorization_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'statements' in args_dict.keys()
	assert 'topic' in args_dict.keys()
	assert 'background' in args_dict.keys()
	
	history.append(process_prompt(prompts['entity_categorization_prompt']['system']))
	history.append(process_prompt(prompts['entity_categorization_prompt']['user'],args_dict))
	return history

def get_edge_reificiation_prompt(args_dict, history=[]):
	# Required args node_name, node_description, existing summary

	assert 'statement' in args_dict.keys()
	assert 'background' in args_dict.keys()
	
	history.append(process_prompt(prompts['edge_reification_prompt']['system']))
	history.append(process_prompt(prompts['edge_reification_prompt']['user'],args_dict))
	return history

def get_text_decomposition_prompt(args_dict,history=[]):
	# topics, text
	assert 'other_topics' in args_dict.keys()
	assert 'topics' in args_dict.keys()
	assert 'text' in args_dict.keys()

	history.append(process_prompt(prompts['text_decomposition_prompt']['system']))
	history.append(process_prompt(prompts['text_decomposition_prompt']['user'],args_dict))

	return history

def get_pivot_node_strategy_prompt(args_dict,history=[]):
	# topics, text
	assert 'query' in args_dict.keys()
	assert 'topics' in args_dict.keys()

	history.append(process_prompt(prompts['pivot_node_strategy_prompt']['system']))
	history.append(process_prompt(prompts['pivot_node_strategy_prompt']['user'],args_dict))

	return history

def get_cluster_description_prompt(args_dict,history=[]):
	assert 'cluster_descriptions' in args_dict.keys()
	assert 'topic' in args_dict.keys()

	history.append(process_prompt(prompts['cluster_theme_prompt']['system']))
	history.append(process_prompt(prompts['cluster_theme_prompt']['user'],args_dict))

	return history

def get_reasoned_generation_prompt(args_dict,history=[]):

	assert 'user_query' in args_dict.keys()
	assert 'local_context' in args_dict.keys()

	if(len(history)==0):
		history.append(process_prompt(prompts['reasoned_answer_prompt']['system']))
	history.append(process_prompt(prompts['reasoned_answer_prompt']['user'],args_dict))

	print(history[-1]['content'])

	return history

def get_triplet_extraction_prompt(args_dict,history=[]):

	assert 'statements' in args_dict.keys()
	assert 'topics' in args_dict.keys()

	history.append(process_prompt(prompts['triplet_extraction_prompt']['system']))
	history.append(process_prompt(prompts['triplet_extraction_prompt']['user'],args_dict))

	return history

def get_triplet_correction_prompt(args_dict,history=[]):

	assert 'statements' in args_dict.keys()
	assert 'triplets' in args_dict.keys()

	history.append(process_prompt(prompts['recheck_triplets_prompt']['system']))
	history.append(process_prompt(prompts['recheck_triplets_prompt']['user'],args_dict))

	return history

def get_query_evaluation_prompt(args_dict,history=[]):

	assert 'context' in args_dict.keys()
	assert 'query' in args_dict.keys()

	history.append(process_prompt(prompts['query_evaluation_prompt']['system']))
	history.append(process_prompt(prompts['query_evaluation_prompt']['user'],args_dict))

	return history

def get_context_sufficiency_check_prompt(args_dict,history=[]):

	assert 'context' in args_dict.keys()
	assert 'query' in args_dict.keys()

	history.append(process_prompt(prompts['context_sufficiency_check_prompt']['system']))
	history.append(process_prompt(prompts['context_sufficiency_check_prompt']['user'],args_dict))

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

def get_reordering_prompt(args_dict,history=[]):

	assert 'phases' in args_dict.keys()
	assert 'outline' in args_dict.keys()

	if(len(history)==0):
		history.append(process_prompt(prompts['plot_reordering_prompt']['system']))
	history.append(process_prompt(prompts['plot_reordering_prompt']['user'],args_dict))

	return history

def get_chain_of_thought_regex():
	reasoning_structure_regex = (
    r"<scratchpad>\n[\s\S]*?\n</scratchpad>\n+"
    r"Final Answer:\s*[\s\S]+"
	)

	return reasoning_structure_regex
'''