import chainlit as cl
from utils.datastore_utils import DatastoreUtilities
import json
import plotly.graph_objects as go
import networkx as nx

with open('config.json', "r", encoding="utf-8") as f:
        config = json.load(f)

@cl.on_chat_start
async def start():
    # Generate the plotly figure
    du = DatastoreUtilities(config,project_name='demo',current_graph='lore',is_core_graph=True)
    du.load_embedding_model()

    context, nodes, edges = du.get_graph_rag_context("What does Rocky look like?",strategy='explore_neighborhood',categories=['plot'],n_hops=2)
    fig = du.create_network_figure(nodes,target_edges=edges)

    # Embed the figure in Chainlit using cl.Plotly
    element = cl.Plotly(name="network_graph", figure=fig, display="inline")

    await cl.Message(
      content="Here is your interactive NetworkX graph rendered with Plotly:",
      elements=[element],
    ).send()

