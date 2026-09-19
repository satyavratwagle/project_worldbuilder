import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
import json



@cl.on_chat_start
async def on_chat_start():

	test_sents = [f'Test sentence {i}' for i in range(5)]

	items_list = []
	for idx,s in enumerate(test_sents):
		items_list.append({"id":idx,"text":s})

	props = {
			"timeout": 6000,
			"topText": "Found the following topics in the text!",
			"Title": "Select topics to track!",
			"items": items_list}

	checklist_element = cl.CustomElement(
		name="AddToGraph",
		props=props
	)

	element_msg = cl.AskElementMessage(
		content="Test Message",
		element=checklist_element
	)

	cl.user_session.set("checklist_element",checklist_element)
	cl.user_session.set("checklist_message",element_msg)
	# 3. Send the component attached to a chat message
	selection_response = await element_msg.send()
	print('Selection :',selection_response)
	