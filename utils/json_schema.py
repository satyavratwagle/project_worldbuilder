from pydantic import create_model, Field

# If Character, Location, Faction, Artifact, Event, Era
pydantic_descriptions = dict()

# Character
def get_pydantic_descriptions(role,topic,subject):

	if(topic=='summary'): return f"A 1-2 sentence summary of the provided scene. You MUST include the time of the day, the date and the location of the scene. If context is provided, DO NOT include it in the summary."

	# Character
	if(role=='character'):
		if(topic=='overview'):return f"A 1-2 sentence summary of all information known about {subject}. If no information is found, respond with 'None'."
		if(topic=='appearance'):return f"Brief description of the physical appearance of {subject}. If no information is found, respond with 'None'."
		elif(topic=='personality'):return f"Brief description of the personality traits shown by {subject}. If no information is found, respond with 'None'."
		elif(topic=='abilities'):return f"Brief description of extraordinary abilities shown by {subject}. If no information is found, respond with 'None'."
		elif(topic=='backstory'):return f"Information about the past of {subject}. If no information is found, respond with 'None'."
		elif(topic=='relationships'):return f"Nature of the relationships of {subject} with others. If no information is found, respond with 'None'."

	# Location
	elif(role=='location'):
		if(topic=='location'):return f"Description of the physical location of {subject} in the world. If no information is found, respond with 'None'."
		elif(topic=='architecture'):return f"Description of the buildings within {subject} if any. If no information is found, respond with 'None'."
		elif(topic=='geography'):return f"Description of the geographical layout of the area around {subject} if given. If no information is found, respond with 'None'."
		elif(topic=='demographics'):return f"Description of the people living in {subject} if any. If no information is found, respond with 'None'."
		elif(topic=='government'):return f"Information about the rulers of {subject} if any. If no information is found, respond with 'None'."
		elif(topic=='history'):return f"Information about the history of {subject} if given. If no information is found, respond with 'None'."

	# Faction
	if(role=='faction'):
		if(topic=='structure'):return f"Description of the organizational structure of {subject}. If no information is found, respond with 'None'."
		elif(topic=='ideology'):return f"Description of the ideology followed by members of {subject} If no information is found, respond with 'None'."
		elif(topic=='influence'):return f"Description of the impact of {subject} on the world. If no information is found, respond with 'None'."
		elif(topic=='history'):return f"Information about the history of {subject}. If no information is found, respond with 'None'."

	# Artifact
	if(role=='artifact'):
		if(topic=='rules'):return f"Description of the rules related to how {subject} operates. If no information is found, respond with 'None'."
		elif(topic=='applications'):return f"Description of the areas of application of {subject}. If no information is found, respond with 'None'."
		elif(topic=='variations'):return f"Description of alternative versions of {subject}. If no information is found, respond with 'None'."
		elif(topic=='mythos'):return f"History and mythos related to {subject}. If no information is found, respond with 'None'."

	# Event
	if(role=='event'):
		if(topic=='duration'):return f"The duration of {subject} in a 'from / to' format,if specified. If no information is found, respond with 'None'."
		elif(topic=='background'):return f"Description of past events, if any, that led up to {subject},if specified. If no information is found, respond with 'None'."
		elif(topic=='process'):return f"Description of {subject} in the present. If no information is found, respond with 'None'."
		elif(topic=='aftermath'): return f"Description of future impact of {subject}, if specified. If no information is found, respond with 'None'."


def create_character_schema():
	schema = dict()
	schema['overview'] = []
	schema['appearance'] = []
	schema['personality'] = []
	schema['abilities'] = []
	schema['backstory'] = []
	schema['relationships'] = []
	schema['misc'] = []

	return schema

def create_location_schema():

	schema = dict()
	schema['overview'] = []
	schema['location'] = []
	schema['architecture'] = []
	schema['geography'] = []
	schema['demographics'] = []
	schema['government'] = []
	schema['history'] = []
	schema['misc'] = []

	return schema

def create_faction_schema():

	schema = dict()
	schema['overview'] = []
	schema['structure'] = []
	schema['ideology'] = []
	schema['influence'] = []
	schema['history'] = []
	schema['misc'] = []

	return schema

def create_artifact_schema():

	schema = dict()
	schema['overview'] = []
	schema['rules'] = []
	schema['applications'] = []
	schema['variations'] = []
	schema['mythos'] = []
	schema['misc'] = []

	return schema

def create_event_schema():

	schema = dict()
	schema['overview'] = []
	schema['duration'] = []
	schema['background'] = []
	schema['process'] = []
	schema['aftermath'] = []
	schema['misc'] = []

	return schema

def create_dictionary_schema():

	schema = dict()
	schema['overview'] = []

	return schema

def get_schema(label):

	label = label.lower()
	if(label=='character'):
		return create_character_schema()
	elif(label=='location'):
		return create_location_schema()
	elif(label=='faction'):
		return create_faction_schema()
	elif(label=='artifact'):
		return create_artifact_schema()
	elif(label=='event'):
		return create_event_schema()
	elif(label=='definition'):
		return create_dictionary_schema()
	else:
		print('No Matching Schema Found!')

def get_pydantic_schema(schema_name,role,subject,full=True):
	schema = get_schema(role)

	field_definitions = {}
	field_definitions['summary'] = (str,Field(description=get_pydantic_descriptions(role,'summary',subject)))
	
	if(full):
		for key, topic in schema.items():
			if(not(key=='misc')):
				val_type = type(topic)
				try:
					field_definitions[key] = (str,Field(description=get_pydantic_descriptions(role,topic,subject)))
				except:
					pass
		
	return create_model(schema_name, **field_definitions)

