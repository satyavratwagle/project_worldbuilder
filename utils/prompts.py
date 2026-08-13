import re

def get_decomposition_prompt(role='system',history=None,user_input='',context=''):

	if(role=='system'):
		prompt = {"role": "system", 
				"content": ("You are an advanced query analysis assistant."
        				"Given a proposition, your job is to list individual, atomic assumptions made by the proposition."
        				"Do not assume any context other than that provided by the user."
        				"Each assumption must be brief and specific."
        				"Output the list of assumptions in bullet points.")}
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
												"followed by a brief answer based on your reasoning in the 'answer' field."
												"Do NOT include any conversational preamble or text. "
												"Your response must start immediately with the character '{'.")}
	elif(role=='user'):
		prompt = {"role": "user", "content": f"User Query: {user_input}\n\nLocal Context :\n{context_text}"}

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

def isolate_scene_element(role='system', history=None, text='',entity='',subjects=['overview']):

	if(role=='system'):
		'''prompt = {"role": "system", "content": (f"Your job is to provide a brief description of {entity}, which is a {aspect} in the context of the given scene. "
												f"Your response must contain ONLY information about {entity} in the context of the given scene."
												f"You must respond with a description of {entity} in the specified response format. "
												f'\n--- TEXT TO ANALYZE ---\n{text.strip()}\n--- END OF TEXT ---')
												}'''
		prompt = {"role":"system","content":(f'Analyze the following scene text and briefly describe all required topics, strictly adhering to the schema definitions.'
												"Do NOT include any conversational preamble or text. "
                								"Your response must start immediately with the character '{'.")}
	elif(role=='user'):
		prompt = {"role":"user","content":(f'Describe the following topics related to {entity} : {", ".join(subjects)} f\n--- TEXT TO ANALYZE ---\n{text.strip()}\n--- END OF TEXT ---\n')}

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