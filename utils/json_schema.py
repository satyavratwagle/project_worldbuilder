# If Character, Location, Faction, Artifact, Event, Era

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

