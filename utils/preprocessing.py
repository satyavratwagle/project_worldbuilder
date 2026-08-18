import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModelForSequenceClassification,AutoTokenizer
import torch

# 1. Define the target JSON structure via Pydantic
'''class RelationTriple(BaseModel):
    subject: str = Field(description="The source entity, e.g., character, faction, or artifact.")
    relation: str = Field(description="The relational verb or connection, formatted in uppercase.")
    object: str = Field(description="The target entity being interacted with.")

class KnowledgeGraphSchema(BaseModel):
    triples: List[RelationTriple]'''

# 2. Load Llama-3.2-3B-Instruct with outlines
#model_id = "meta-llama/Llama-3.2-3B-Instruct"

'''model = outlines.from_transformers(
    AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="mps"),
    AutoTokenizer.from_pretrained(model_id)
)'''

def get_entailment_probs(premise,hypothesis):

    nli_tokenizer = AutoTokenizer.from_pretrained("tasksource/ModernBERT-base-nli")
    nli_model = AutoModelForSequenceClassification.from_pretrained("tasksource/ModernBERT-base-nli")

    #premise = f"Context: {context}\nPremise: {premise}"
    #hypothesis = f"hypothesis: {hypothesis}"
    inputs = nli_tokenizer(premise, hypothesis, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = nli_model(**inputs).logits
    # Label index 1 is typically 'entailment' in cross-encoder models
    probs = torch.softmax(logits, dim=1).squeeze()
    return probs[0],probs[1],probs[2] # Contradiction, Entailment, Neutral

def get_similarity_matrix(sentences):
    # Get the similarity matrix of sentence embeddings

    # Input : sentences (list)
    # Output : similarity_matrix (matrix)

    # Compute pairwise similarity and select consensus candidate

    embed_model = SentenceTransformer("BAAI/bge-m3")
    embeddings = embed_model.encode(sentences)
    similarity_matrix = cosine_similarity(embeddings)

    return similarity_matrix

def get_best_summary(summaries):
    # Given a list of summaries, pick the one that best summarizes the text.

    # Input : summaries (list)
    # Output : best_summary (str)

    similarity_matrix = get_similarity_matrix(summaries)
    centroid_idx = np.argmax(np.mean(similarity_matrix, axis=1))
    best_summary = summaries[centroid_idx]

    return best_summary

def get_corpus(corpus_path='../data/test_corpus.txt'):
    # Retrieve a corpus from a text file

    # Input : corpus_path (string)
    # Output : corpus_text (string)

    corpus = []
    with open(corpus_path,'r') as file:
        for row in file.readlines():
            if(len(row.strip().strip('\n'))>0):
                corpus.append(row.replace('\n',''))

    corpus_text = "\n".join(corpus)
    corpus_text = corpus_text.replace('``','"')

    return corpus_text

def get_chunks(corpus_text,chunk_size=400):
    # Split a corpus into chunks

    # input : corpus_text (string)
    # Output : chunks (list)

    sentence_pattern = r"(?<=[.!?])\s+"
    sentences = re.split(sentence_pattern, corpus_text.strip())

    #sentences = corpus_text.split('\n')

    chunks = []
    new_sentence = ""
    for s in sentences:
        new_sentence += s

        if(len(new_sentence)>chunk_size):
            chunks.append(new_sentence)
            new_sentence = ""

    if(len(new_sentence)>0):
        chunks.append(new_sentence)

    return chunks


def get_sentences(corpus_text):
    # Split a corpus into sentences.

    # Input : corpus (string)
    # Output : sentences (list)

    corpus_text = re.sub(r'\.+','.',corpus_text)
    sentences = corpus_text.split(".")

    merged_sentences = []
    dialog_begun = False
    for idx in range(len(sentences)):
        if('"' in sentences[idx]):
            if(not(dialog_begun)):
                # A new dialog has begin
                new_sentence = sentences[idx]+'.'
                dialog_begun = True
            else:
                # Dialog has ended.
                new_sentence += sentences[idx]
                merged_sentences.append(new_sentence)
                dialog_begun = False
            
        else:
            if(not(dialog_begun)):
                # Normal sentence
                merged_sentences.append(sentences[idx])
                #print('New:',sentences[idx])
            else:
                # A sentence within a dialog
                new_sentence += sentences[idx]+'.'

    return merged_sentences

def semantic_scene_change_detection(sentences,threshold=0.45):
    # Check if the Scene has changed.

    # Input : sentences (list)
    # Output : clusters (list)

    embed_model = SentenceTransformer("BAAI/bge-m3")

    sentence_embeddings = embed_model.encode(sentences)

    # Calculate cosine similarity between consecutive paragraph pairs
    similarities = []
    for i in range(len(sentence_embeddings) - 1):
        sim = cosine_similarity([sentence_embeddings[i]], [sentence_embeddings[i+1]])[0][0]
        similarities.append(sim)

    '''for idx in range(len(similarities)):
        print(similarities[idx])
        print(len(sentences[idx]),sentences[idx])
        print(len(sentences[idx+1]),sentences[idx+1])
        print()'''


    # Cluster sentences based on semantic distance
    threshold = threshold

    clusters = []
    current_cluster = []
    for idx in range(len(similarities)):
        if(similarities[idx]>threshold):
            current_cluster.append(sentences[idx])
        else:
            clusters.append(current_cluster)
            current_cluster = [sentences[idx]]

    # Combine smaller outlier clusters
    long_clusters = []
    new_cluster = []
    for idx in range(len(clusters)):
        if(sum(len(sentence) for sentence in clusters[idx])<50):
            new_cluster += clusters[idx]
        else:
            if(sum(len(sentence) for sentence in new_cluster)>40):
                long_clusters.append('.'.join(new_cluster))
                new_cluster = []
            else:
                new_cluster = []
                long_clusters.append('.'.join(clusters[idx]))

    return long_clusters

def extract_dialogs(corpus):
    # Extract dialogs from corpus

    # Input : corpus (string)
    # Output : dialogs (list)

    dialogs = re.findall(r'\`\`[^\"]+\"[^\.\`]+\.',corpus_text)

    for idx in range(len(dialogs)):
        dialogs[idx] = dialogs[idx].replace("``",'"')


    return dialogs