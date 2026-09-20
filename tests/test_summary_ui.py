import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
import json

items_list = []
items_list.append({"id":0,"node":"node","node_name":"node_name","text":"this is a big summary that makes it longer than your usual summary. So this will spill onto multiple lines. After all our summary is our summary none of your summary.","wiki_key":"wiki_key"})

@cl.on_chat_start
async def on_chat_start():
	
	props = {
			"timeout": 6000,
			"topText": "TopText",
			"Title": "Title",
			"items": items_list}

	summary_element = cl.CustomElement(
		name="SummarySelectionElement",
		props=props
	)

	element_msg = cl.AskElementMessage(
		content="UI Message",
		element=summary_element
	)

	selection_response = await element_msg.send()
	print('Selection :',selection_response)