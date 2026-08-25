import re

def get_decomposition_prompt(role='system',history=None,user_input='',context=''):

	if(role=='system'):
		prompt = {"role": "system", 
				"content": ("You are an advanced query analysis assistant."
        				"Given a proposition, your job is to list individual, atomic assumptions made by the proposition."
        				"Do not assume any context other than that provided by the user."
        				"Do not include assumptions that are confirmed by the context."
        				"Each assumption must be brief and specific."
        				"Your response must follow the provided schema.")}
	elif(role=='user'):
		prompt = {"role": "user", 
				"content": f"Proposition: {user_input}\n\nContext:{context}"}

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def get_generation_prompt(role='system', history=None, user_input='', context_text=''):

	if(role=='system'):
		prompt = {"role": "system", "content": (f"You are a helpful assistant. You must answer the user query using ONLY the local context."
												"Always output your reasoning in verbose form in the 'scratchpad' field,"
												"followed by a descriptive answer based on your reasoning in the 'answer' field."
												"Do NOT include any conversational preamble or text. "
												"Your response must start immediately with the character '{'.")}
	elif(role=='user'):
		prompt = {"role": "user", "content": f"User Query: {user_input}\n\nLocal Context :\n{context_text}"}

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def get_timeplace_prompt(scene='', qa_dict=None):

	prompt_history = []	

	system_prompt = {"role": "system", "content": (f"You are a helpful assistant. You must answer the user query using ONLY the local context."
											"Do NOT include any conversational preamble or text. "
											"Your response must follow to provided schema.")}
	prompt_history.append(system_prompt)

	user_prompt_content = 'Given that the units of measuring time are FMC or Crossings, answer the following questions\n'
	for key in qa_dict.keys():
		user_prompt_content += f'{key}\nYour options are : {qa_dict[key]}\n'
	user_prompt_content+= f'\n---SCENE BEGINS---\n{scene}\n---SCENE ENDS---\n'
	user_prompt = {"role": "user", "content": user_prompt_content}

	prompt_history.append(user_prompt)

	return prompt_history

def get_summarization_prompt(role='system', history=None, entity='',type='',subject='',text=''):

	if(role=='system'):
		prompt = {"role": "system", "content": (f"Your job is to summarize the text given by the user."
												"Use only the given text to summarize. Do NOT invent or extrapolate information."
												"Do NOT include any conversational preamble or text."
												"Answer in strictly 2 or 3 sentences.")}
	elif(role=='user'):
		prompt = {"role": "user", "content": (f"The following text contains the descriptions of the {subject} of {entity}, a {type} in a fictional world.\n"
												f"Descriptions of {subject} of {entity} are described as they evolve across time, separated by ';'.\n"
												f"Summarize the most consistent descriptions of {subject} of {entity} in the given text.\n"
												"Do NOT include scene numbers or references to the scenes in your response."
												f"\n\n--- TEXT TO SUMMARIZE BEGINS ---{text}--- TEXT TO SUMMARIZE ENDS ---")}

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def get_context_retrieval_prompt(role='system', history=None, query='',relevant_entities=''):

	if(role=='system'):
		prompt = {"role": "system", "content": (f"You are a RAG query generator."
												"Your job is to create a lookup command for each of the given Relevant Entities"
												"by isolating the function of each entity in the given User Query."
												"You must respond strictly as a JSON list of strings."
												"You must ONLY use the information available in the User Query to create the commands."
												)}
	elif(role=='user'):
		prompt = {"role": "user", "content": f"Relevant Entities: {relevant_entities}\nUser Query: {query}"}
		print(prompt)

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def get_gist_prompt(role='system', history=None, text='',tracked_entities=[]):

	if(role=='system'):
		'''prompt = {"role": "system", "content": (f"Your job is to specify the topics given in the response format that are involved in the scene. "
												"Only include topics with significant presence in the scene. "
												"If no topic is found, respond with 'None'. "
												"Characters specified must be living beings. "
												"Events specified must be significant. "
												"You must ONLY use the context provided to specify the topics. "
												f"\n\nContext:{context}\n\nScene:{text}"
												)}'''
		prompt = {"role":"system","content":(f'Analyze the following scene text and extract allrequired information accurately, strictly adhering to the schema definitions.'
												"Answer each field in strictly 1 or 2 sentences."
												"Do NOT include any conversational preamble or text. "
                								"Your response must start immediately with the character '{'."
												)}
	elif(role=='user'):
		if(len(tracked_entities)>0):
			prompt = {"role":"user","content":(f'Currently tracked entities are : {', '.join(tracked_entities)}\n\n--- TEXT TO ANALYZE ---\n{text.strip()}\n--- END OF TEXT ---\n')}
		else:
			prompt = {"role":"user","content":(f'\n--- TEXT TO ANALYZE ---\n{text.strip()}\n--- END OF TEXT ---\n')}

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def isolate_scene_element(role='system', history=None, text='',entity='',subjects=['overview'], aliases=[],scene_context=None,rag_context=None,full=True):

	if(role=='system'):
		'''prompt = {"role": "system", "content": (f"Your job is to provide a brief description of {entity}, which is a {aspect} in the context of the given scene. "
												f"Your response must contain ONLY information about {entity} in the context of the given scene."
												f"You must respond with a description of {entity} in the specified response format. "
												f'\n--- TEXT TO ANALYZE ---\n{text.strip()}\n--- END OF TEXT ---')
												}'''
		prompt = {"role":"system","content":(f'Analyze the following scene text and briefly describe all required topics, strictly adhering to the schema definitions. '
												"Answer each field in strictly 1 or 2 sentences. "
												"Do NOT include any conversational preamble or text. "
												"You must use only the scene text and supplementary information provided by the user when describing the topics. Do NOT invent or extrapolate information. "
                								"Your response must start immediately with the character '{'.")}
	elif(role=='user'):

		if(full):
			subjects.remove('summary')
			character_prompt = f'Use the current scene to describe the following topics related to {entity} : {", ".join(subjects)}; also provide a brief summary of the scene.'
		else:
			character_prompt = f'Use the current scene to provide a brief summary of the scene.'

		if(scene_context):
			prompt = {"role":"user","content":(character_prompt+f'Do not use the context from previous scene directly in your responses. '
												f'You may only use relevant supplementary information in your responses.'
												f'\n {entity} is also referred to as {','.join(aliases)}.'
												f'\n In all fields except "summary", your response should contain ONLY information about {entity}.'
												f'In all fields except "summary", Do not include information about anything else.'
												f'\n--- SUPPLEMENTARY INFORMATION ---\n{rag_context.strip()}\n--- END OF SUPPLEMENTARY INFORMATION---\n\n'
												f'\n--- CONTEXT FROM PREVIOUS SCENE ---\n{scene_context.strip()}\n--- END OF CONTEXT FROM PREVIOUS SCENE---\n\n'
												f'\n--- CURRENT SCENE ---\n{text.strip()}\n--- END OF CURRENT SCENE ---\n')}
		else:
			prompt = {"role":"user","content":(character_prompt+f'\n {entity} is also referred to as {','.join(aliases)}.'
												f'\n Your response should contain ONLY information about {entity}.'
												f'Do not include information about anything else.'
												f'\n--- SUPPLEMENTARY INFORMATION ---\n{rag_context.strip()}\n--- END OF SUPPLEMENTARY INFORMATION---\n\n'
												f'\n--- CURRENT SCENE ---\n{text.strip()}\n--- END OF CURRENT SCENE ---\n')}

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def get_chain_of_thought_regex():
	reasoning_structure_regex = (
    r"<scratchpad>\n[\s\S]*?\n</scratchpad>\n+"
    r"Final Answer:\s*[\s\S]+"
	)

	return reasoning_structure_regex