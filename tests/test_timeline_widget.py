import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
import json

items_list = []
items_list.append({"id":0,"node":"node","node_name":"node_name","text":"this is a big summary that makes it longer than your usual summary. So this will spill onto multiple lines. After all our summary is our summary none of your summary.","wiki_key":"wiki_key"})

@cl.on_chat_start
async def on_chat_start():
	
	props = {
				"steps":[
							{
								"id":1,
								"title":"First Step",
								"subtitle":"Subtitle 1"
							},
							{
								"id":2,
								"title":"Second Step",
								"subtitle":"Subtitle 2"
							},
							{
								"id":3,
								"title":"Third Step",
								"subtitle":"Subtitle 3"
							}
						]
			}

	timeline_widget = cl.CustomElement(
		name="TimelineWidget",
		props=props
	)

	cl.user_session.set("timeline_widget", timeline_widget)

@cl.on_message
async def on_message(user_message: cl.Message):

	timeline_widget = cl.user_session.get("timeline_widget")

	timeline_widget.props['steps'].append({"id":4,"title":user_message.content,"subtitle":"User"})

	print(timeline_widget)

	element_msg = cl.Message(
		content="UI Message",
		elements=[timeline_widget]
	)

	selection_response = await element_msg.send()
	print('Selection :',selection_response)



