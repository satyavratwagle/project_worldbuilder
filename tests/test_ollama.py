import ollama
import json
import ast
from utils.datastore_utils import DatastoreUtilities

with open('config.json', "r", encoding="utf-8") as f:
            config = json.load(f)

du = DatastoreUtilities(config)

def search_external_database(topics: list) -> str:
    topics = ast.literal_eval(topics)
    return "\n".join([du.get_node_summary(topic) for topic in topics])

def find_connection_between(topic1:str,topic2:str) -> str:
    path = du.find_path(topic1,topic2)

    if(path):
        path_descriptions = []
        for node in path:
            path_descriptions.append(du.get_node_summary(node))
        return " -> ".join(path_descriptions)
    else:
        return f"{du.get_node_summary(topic1)} {du.get_node_summary(topic2)} No connection exists between these entities."

def not_applicable():

    return ""

tools_json = [
    {
        'type': 'function',
        'function': {
            'name': 'search_external_database',
            'description': "Search the external database for information about unknown topics, facts, or entities. Returns: External information about the specified topics.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'topics': {'type': 'list', 'description': 'The names of the unknown topics to query.'}
                },
                'required': ['topics']
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "not_applicable",
            "description": "Call this ONLY if the user's request can be answered using your internal knowledge and does not require external data or tools.",
            "parameters": {
              "type": "object",
              "properties": {}
            }
        }
    },
    {'type': 'function',
        'function': {
            'name': 'find_connection_between',
            'description': "Search for relationships between unknown topics, facts, or entities. Use only when the user query requests information about the relationship or connection between two entities. Returns: The relationship between the two specified topics.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'topic1': {'type': 'str', 'description': 'The first topic to query.'},
                    'topic2': {'type': 'str', 'description': 'The second topic to query.'}
                },
                'required': ['topic1','topic2']
            }
        }
    }
]

available_tools = {
    "search_external_database":search_external_database,
    "not_applicable":not_applicable,
    "find_connection_between":find_connection_between
}

messages = [
        {
            'role': 'system', 
            'content': (
                'You are a helpful data querying assistant for fictional worldbuilding.'
                
            )
        },
        {
            'role': 'user', 
            'content': 'What is the connection between Mahamun and Citta?'}]

response = ollama.chat(
    model='llama3.1:8b',
    messages=messages,
    tools=tools_json,
    options={
        'temperature': 0.25,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
        'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
    }
)

print(response)

if response.message.tool_calls:
    for tool in response.message.tool_calls:
        print(f"Tool called: {tool.function.name}")
        print(f"Arguments: {tool.function.arguments}")

        function_name = tool.function.name
        function_args = tool.function.arguments

        if function_name in available_tools:
            tool_to_call = available_tools[function_name]
            tool_output = tool_to_call(**function_args)

            print(f"   - Tool Execution Result: {tool_output}")

            messages.append({
                    "role": "tool",
                    "content": tool_output,
                })
                
            # 5. Second API call: Send history back so the model can read the tool output and reply to user
            print("\nSending tool output back to the model for final response...")
            final_response_stream = ollama.chat(
                model='llama3.1:8b',
                messages=messages,
                options={
                            'temperature': 0.7,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
                            'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
                        },
                stream = True
            )

            for chunk in final_response_stream:
                print(chunk.message.content, end='', flush=True)