import chainlit as cl
#from utils.datastore_utils import DatastoreUtilities
import json

items_list = []
items_list.append({"id":0,"node":"node","node_name":"node_name","text":"this is a big summary that makes it longer than your usual summary. So this will spill onto multiple lines. After all our summary is our summary none of your summary.","wiki_key":"wiki_key"})

@cl.on_chat_start
async def on_chat_start():
	
	props = {"items": []}
	

	props['items'].append({
							"id":str(0),
							"name":"Topic Name",
							"aliases":["Alias 1","Alias 2"],
							"type":"Topic Type",
							"tag":"Topic Tag",
							"reasoning": "Sample reasoning for the topic type and subcategory."
						})
		
	print(items_list)
	

	checklist_element = cl.CustomElement(
		name="SelectToTrack",
		props=props
	)

	element_msg = cl.AskElementMessage(
		content="Message Here.",
		element=checklist_element,
		timeout=360
	)
	# 3. Send the component attached to a chat message
	selection_response = await element_msg.send()
	print(selection_response)