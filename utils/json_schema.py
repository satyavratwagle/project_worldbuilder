from pydantic import create_model, Field

# If Character, Location, Faction, Artifact, Event, Era
pydantic_descriptions = dict()

# Character
pydantic_descriptions['appearance'] = "The physical appearance of the character."
pydantic_descriptions['personality'] = "The personality traits shown by the character."
pydantic_descriptions['abilities'] = "Extraordinary abilities shown by the character."
pydantic_descriptions['backstory'] = "Information about the character's past."
pydantic_descriptions['relationships'] = "Nature of the relationships of the character with others."

# Location
pydantic_descriptions['location'] = "Physical location in the world."
pydantic_descriptions['architecture'] = "Description of the buildings at the location if any."
pydantic_descriptions['geography'] = "Geographical layout of the location."
pydantic_descriptions['demographics'] = "Description of the people living in the location if any."
pydantic_descriptions['government'] = "Information about the rulers of the location if any."
pydantic_descriptions['history'] = "Information about the history of the location."

# Faction
pydantic_descriptions['structure'] = "The organizational structure of the faction."
pydantic_descriptions['ideology'] = "The ideology followed by members of the faction"
pydantic_descriptions['influence'] = "The impact of the organization on the world."
pydantic_descriptions['history'] = "Information about the history of the faction."

# Artifact
pydantic_descriptions['rules'] = "Rules related to how the artifact operates."
pydantic_descriptions['applications'] = "Specified areas of application of the artifact."
pydantic_descriptions['variations'] = "Alternative versions of the artifact."
pydantic_descriptions['mythos'] = "History and mythos related to the artifact."

# Event
pydantic_descriptions['background'] = "Past events, if any, that led up to the current event."
pydantic_descriptions['process'] = "What is happening in the present."
pydantic_descriptions['aftermath'] = "Future impact of the event, if specified."


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

def get_pydantic_schema(schema_name,label):

	schema = get_schema(label)

	field_definitions = {}
	
	for key, value in schema.items():
		if(not(key=='overview') and not(key=='misc')):
			val_type = type(value)
			try:
				field_definitions[key] = (val_type,Field(description=pydantic_descriptions[key]))
			except:
				pass
		
	return create_model(schema_name, **field_definitions)

