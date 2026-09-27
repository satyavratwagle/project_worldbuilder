# Quill

> Quill is a worldbuilding and brainstorming assistant for novel writers and authors. It turns all your ideas into a Wikipedia for your world! 

Quill is able to:
1. Ingest your ideas as natural language and convert them into interconnected topics, complete with topic summaries and categories.
2. Answer questions related to all of your ideas, collecting information about all related topics before answering.
3. Brainstorm about new ideas based on your old ones, extrapolating ideas to find interesting new connections and relationships.

# Initial Setup

## 1. Set up your Folder Structure
Your project structure should be as follows.
```
- ROOT_FOLDER
	- data
		- graphs
		- faiss_store
	- project_worldbuilder
		- app.py
		- utils
		- test 
```

## 2. Install Requirements
Install all packages in `requirements.txt`
- The UI and asynchronous operations are handled through `chainlit`
- The language model is served through `ollama` 
- The knowledge graph back-end is handled through `networkx`

## 3. Set Up Configuration Files
Create `ROOT_FOLDER/project_worldbuilder/config.json`. Quill looks for the `config.json` file within the `project_worldbuilder` folder for conifguration data. 

Based on the set up in step 1; the recommended set up for `config.json` is as follows:
```
"data_dir":"ROOT_FOLDER/data",
"model_id": "Llama-3.1-8B-Instruct-quantized-4b",
"graph_dir":"ROOT_FOLDER/data/graphs",
"kg_to_ds_map":"ROOT_FOLDER/data/kg2ds.json",
"text_embedding_model":"BAAI/bge-m3",
"ner_extraction_model":"knowledgator/gliner-relex-large-v1.0",
"zero_shot_classification_model":"knowledgator/gliclass-modern-base-v3.0",
"nli_model":"cross-encoder/nli-deberta-v3-large"
```
**NOTE :** Quill has not been tested with any other LLM, NER, NLI, or Zero-shot classification models. These values can be changed, but they may require some changes to be made within the corresponding scripts.

# Running Quill
Run `chainlit run app.py` in your command line. Quill should start up in your default browser after loading up the models.

## Functions
1. By default, Quill functions as a question-answering assistant for your knowledge graph. You can ask any query to Quill, and it will determine the theme of your question, the filters to put on the knowledge graph, as well as the strategy to use to extract information from the knowledge graph.
2. To add an idea to Quill, input `/ideate` before your idea. Quill will then extract relevant topics and information from the idea to add to the knowledge graph.
3. If you are happy with the information Quill has extracted, enter `/update` to permanently add the extracted information to Quill's internal database.

