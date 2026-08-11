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
												"followed by a brief answer based on your reasoning in the 'answer' field.")}
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

def get_gist_prompt(role='system', history=None, text='',context='No context available',subjects=['characters','location','events']):

	if(role=='system'):
		prompt = {"role": "system", "content": (f"Your job is to summarize the given scene in the given format. "
												"The summary must specify the following subjects involved in the scene. "
												f"\nSubjects : {', '.join(subjects[:-1])} and {subjects[-1]}.\n"
												"If no subject is found, respond with 'None'. "
												"You must ONLY use the context provided to summarize. "
												f"\n\nContext:{context}\n\nScene:{text}"
												)}
	elif(role=='user'):
		prompt = {"role": "user", "content": f"Relevant Entities: {relevant_entities}\nUser Query: {query}"}
		print(prompt)

	if(history):
		history.append(prompt)
		return history
	else:
		return [prompt]

def isolate_scene_element(role='system', history=None, text='',entity='',aspect='',subjects=['overview']):

	if(role=='system'):
		prompt = {"role": "system", "content": (f"Your job is to elaborate upon {entity}, which is a {aspect} in the context of the given scene. "
												f"Your response must contain ONLY information about {entity} in the context of the given scene."
												f"You must describe the following features of {entity} in the scene : {', '.join(subjects)}\n "
												f"\n\nScene:{text}"
												)}
	elif(role=='user'):
		prompt = {"role": "user", "content": f"Relevant Entities: {relevant_entities}\nUser Query: {query}"}

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