import networkx as nx
import torch
import math
import plotly.graph_objects as go
import textwrap

class KnowledgeTree():

    def __init__(self):

        self.G = nx.DiGraph()
        self.node_idx = -1

    def add_node(self,text,type, layer, context,topics, answer):

        self.node_idx += 1
        self.G.add_node(self.node_idx,text=text,type=type, layer=layer, context=context, topics=topics, answer = answer)

        return self.node_idx

    def add_answer_to_context(self,node_id,answer,confidence):
        self.G.nodes[node_id]['answer'] = answer
        if(not(node_id)==0):
            self.G.nodes[node_id]['context'].append(f"[IMPORTANT QUESTION]\nQuestion : {self.G.nodes[node_id]['text']}\nPredicted Answer : {self.G.nodes[node_id]['answer']} (Confidence={confidence})") 

    def add_edge(self,head,tail):
        self.G.add_edge(head,tail)

    def check_for_new_topics(self,node_id):
        print(f"{self.G.nodes[node_id]['text']} : {len(self.G.nodes[node_id]['topics'])}")
        return (len(self.G.nodes[node_id]['topics'])>0)


    def get_context_by_node(self,node_id):

        path = nx.shortest_path(self.G, source=0, target=node_id)

        context_list = []
        topics_to_ignore = []
        for node in path:
            context_list += self.G.nodes[node]['context']
            topics_to_ignore += self.G.nodes[node]['topics']

        topics_to_ignore = list(set(topics_to_ignore))

        return context_list, topics_to_ignore

    def get_topics_to_explore(self,previous_node):
        print(self.G.nodes[previous_node])
        return self.G.nodes[previous_node]['topics']
        
    def get_query_by_node(self,node_id):
        if(node_id in self.G.nodes()):
            return self.G.nodes[node_id]['text']

    def get_topics_from_branch(self,node_id):

        path = nx.shortest_path(self.G, source=0, target=node_id)

        branch_topics = []
        if(len(path)>1):
            for idx in range(len(path)-1):
                source = path[idx]
                target = path[idx+1]
                edge = self.G.get_edge_data(source,target)
                branch_topics += edge['topics']

        branch_topics = list(set(branch_topics))

        return branch_topics, path

    def format_branch(self,path,topic_list,topic_lookup):

        context = ""
        if(len(path)>1):
            for idx in range(len(path)-1):
                source = path[idx]
                target = path[idx+1]
                edge = self.G.get_edge_data(source,target)

                context += f"Question : {self.G.nodes[source]['text']}\n\n"
                context += "Information :\n"+"\n\n".join([f"\t[{topic}]\n\tDefinition : {topic_lookup(topic)}" for topic in edge['topics']])
                context += f"\n\nAnswer : {edge['text']}\n\n"

        context += "New Important Information :\n"+"\n\n".join([f"\t[{topic}]\n\tDefinition : {topic_lookup(topic)}" for topic in topic_list])

        return context

    def format_graph(self,filter_func, topic_lookup):

        
        all_text = []
        for node,data in self.G.nodes(data=True):
            for context in data['context']:
                if(not(context in all_text)):
                    all_text.append(context)
        #return all_topics,all_text

        '''
        filtered_text = []
        for idx,q1 in enumerate(all_text):
            if(len(filtered_text)>0):
                cosine_sims = filter_func([q1],filtered_text)[0]
                if(max(cosine_sims)<0.8):
                    filtered_text.append(q1)
            else:
                filtered_text.append(q1)
        '''

        '''
        gathered_information = ""
        gathered_information += "[CONTEXT]\n"+"\n\n".join([f"\t[{topic}]\n\tDefinition : {topic_lookup(topic)}" for topic in all_topics])
        gathered_information += "\n\n[FOLLOW-UP Q/A]\n(Do NOT consider the following to be ground truth.)\n\n"+"\n\n".join(filtered_text)
        '''

        gathered_information = "\n\n".join(all_text)

        return gathered_information

    def get_plotly_graph(self,focus_node=None):

        # Create a subgraph with the target nodes and target edges

        pos = nx.multipartite_layout(self.G, subset_key='layer', align='vertical')
        self.fig_width = 640
        self.fig_height = 640

        annotations = []
        endpoints = []
        edge_text = []
        node_text = []
        node_ids = []
        node_labels = []
        node_colors = []
        border_widths = []
        border_colors = []

        nodes_x,nodes_y     = [],[]
        edges_x, edges_y    = [],[]
        mid_x, mid_y        = [],[]

        for u, v, data in self.G.edges(data=True):
            endpoints.append((u,v))

            x0, y0 = pos[u]
            x1, y1 = pos[v]

            u_pos = (x0*(self.fig_width),y0*(self.fig_height))
            v_pos = (x1*(self.fig_width),y1*(self.fig_height))

            standoff_value = math.dist(u_pos,v_pos)

            edges_x.extend([x0, x1, None])
            edges_y.extend([y0, y1, None])

            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            #edge_text.append("<br>".join(textwrap.wrap(data['text'], width=36)))

            # Add arrowhead annotation for secondary/other edges
            annotations.append(
                dict(
                    x=x1, y=y1,
                    ax=x0, ay=y0,
                    xref="x", yref="y",
                    axref="x", ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowwidth=2.0,
                    arrowcolor="#EEE",
                    standoff = standoff_value
                )
            )

        edge_trace = go.Scatter(
          x=edges_x,
          y=edges_y,
          line=dict(width=2.0, color="#EEE"),
          hoverinfo="none",
          mode="lines"
        )

        hover_trace = go.Scatter(
          x=mid_x,
          y=mid_y,
          mode="markers",
          hoverinfo="text",
          text=edge_text,
          marker=dict(opacity=0, size=20),
        )

        # Create Node Traces

        marker_size = 20

        for node,data in self.G.nodes(data=True):
            x, y = pos[node]
            nodes_x.append(x)
            nodes_y.append(y)

            formatted_node_text = "<b>Question</b><br>"+"<br>".join(textwrap.wrap(data['text'], width=30))+"<br><b>Answer</b><br>"+"<br>".join(textwrap.wrap(f"{data['answer']}", width=30))
            node_text.append(formatted_node_text)
            node_ids.append(node)

            # You can change this later.

            if(focus_node and node==focus_node):
                border_widths.append(5)
                border_colors.append('red')
            else:
                border_widths.append(0)
                border_colors.append("#444444")
            node_colors.append("#EEE")
            node_labels.append("")


        node_trace = go.Scatter(
          x=nodes_x,
          y=nodes_y,
          mode="markers+text",
          textposition="bottom center",
          hoverinfo="text",
          hovertext=node_text,
          text=node_labels,
          textfont=dict(
                color="#EEE",  
                size=marker_size,  
                family=('system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif')
            ),
          marker=dict(
                showscale=False,
                color="#EEE",
                line = dict(
                        width = border_widths,
                        color = border_colors
                    ),
                size=30)
        )

        node_trace.marker.color = node_colors

        fig = go.Figure(
          data=[edge_trace],
          layout=go.Layout(
              showlegend=False,
              hovermode="closest",
              plot_bgcolor="rgba(0,0,0,0)",  
              paper_bgcolor="rgba(0,0,0,0)",  
              annotations=annotations, # <-- Added arrow annotations here
              hoverlabel=dict(
                    bgcolor="#ffffff",  
                    font_color="#000000",  
                    font_size=12,  
                    font_family=(
                        'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",'
                        ' Roboto, Helvetica, Arial, sans-serif'
                    ), 
                    bordercolor="#ffffff",  
                ),
              dragmode="pan",
              xaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False
              ),
              yaxis=dict(
                showgrid=False, zeroline=False, showticklabels=False
              ),
          ),
        )

        fig.update_annotations(startstandoff=marker_size)
        fig.add_trace(hover_trace)
        fig.add_trace(node_trace)

        return fig
        
        

