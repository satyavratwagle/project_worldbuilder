import chainlit as cl
#from utils.datastore_utils import DatastoreUtilities
import json

items_list = []
items_list.append({"id":0,"node":"node","node_name":"node_name","text":"this is a big summary that makes it longer than your usual summary. So this will spill onto multiple lines. After all our summary is our summary none of your summary.","wiki_key":"wiki_key"})

@cl.on_chat_start
async def on_chat_start():
	
	props = {'triplets':[]}
	

	triplet_data = dict()
	triplet_data['id'] = 0
	triplet_data['head'] = ['Source 1','Source 2']
	triplet_data['relation'] = "Sample Relation"
	triplet_data['tail'] = ['Target 1','Target 2']
	triplet_data['base_fact'] = "Base Fact"
	triplet_data['temporal_metadata'] = "Temporal Metadata"

	props['triplets'].append(triplet_data)



	summary_element = cl.CustomElement(
		name="TripletListWidget",
		props=props
	)

	element_msg = cl.AskElementMessage(
		content="UI Message",
		element=summary_element
	)

	selection_response = await element_msg.send()
	print('Selection :',selection_response)

	if(selection_response):
		for key in selection_response.keys():
			if(not(key=='submitted')):
				selected_heads = [selection_response[key]['head'][idx] for idx in [int(key) for key,value in selection_response[key]['head_selection'].items() if value]]
				selected_tails = [selection_response[key]['tail'][idx] for idx in [int(key) for key,value in selection_response[key]['tail_selection'].items() if value]]

				print('Selected Heads :',selected_heads)
				print('Selected Tails :',selected_tails)