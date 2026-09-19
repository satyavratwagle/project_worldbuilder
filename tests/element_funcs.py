import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
import json

items_list = []
items_list.append({"id":"Test ID",
					"label":"Test Label",
					"description":"Test Node Description",
					"defaultChecked": False})

@cl.action_callback("change_node_type")
async def on_change_node_type(action: cl.Action):
	payload = action.payload
	node = payload.get("node")
	
	cl.user_session.set("awaiting_input_label",True)
	#response = await cl.Message(content=f"Choose a new label for {node['label']}!").send()
	updated_label = await cl.AskActionMessage(content="Choose the updated label!",
											actions= [cl.Action(name=label,payload={label:None},label=label) for label in['A','B','C']]).send()

	print(updated_label)

	items_list[0]['description'] = updated_label['name']
	items_list[0]['defaultChecked'] = True
	
	checklist_element = cl.user_session.get("checklist_element")
	checklist_element.props['items'] = items_list
  
	await checklist_element.update()


@cl.on_chat_start
async def on_chat_start():
	cl.user_session.set("awaiting_input_label",False)


@cl.on_message
async def on_message(user_message: cl.Message):

	print('Jere')
	waiting_for_label_update = cl.user_session.get("awaiting_input_label")

	if(waiting_for_label_update):

		print("here!")

		cl.user_session.set("awaiting_input_label",False)

		print(updated_label)


		

	else:
		props = {
				"timeout": 6000,
				"topText": "Found the following topics in the text!",
				"Title": "Select topics to track!",
				"items": items_list}

		checklist_element = cl.CustomElement(
			name="SelectToTrack",
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
		print("HERE!")
		
		print("here x!")