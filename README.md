# Project Worldbuilder

Experiments in using Language Models to assist in novel-writing.

# Initial Setup
1. Install `requirements.txt` (Some packages may be redundant / obsolete.)

# Instructions to Run

1. Set the directory for the FAISS Store in `config.json`
2. To run the web application, use the command `chainlit run app.py`

# Functionalities

## Query Answering
- **Flow :** `Retrieve Context` -> `Use Local Context to Answer Question` -> `Check Context Sufficiency` (TBD)

## Text Analysis
- **Flow :** `Scene Detection` -> `Entity Extraction + Description` -> `Long-Term Dependency Extraction` (TBD)

## 