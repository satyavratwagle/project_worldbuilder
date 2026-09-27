from pydantic import BaseModel, Field
import utils.prompts as prompts
import ollama
from enum import Enum
#from utils.semantic import SemanticTools

class KnownEntities(str, Enum):
    CAUSE = "cause"
    PROCESS = "process; event; action"
    LOCATION = "location; place; position; coordinates"
    PROPERTY = "property; quality; status; tag"
    TIME = "time; duration; era; time period"

class FictionalTriplet(BaseModel):
    source: str = Field(description="Normalized canonical name of the source entity, e.g., 'Prince Elian'")
    relation: str = Field(description="A verbose description of the relation between the source and target.")
    target: str = Field(description="Normalized canonical name of the target entity, e.g., 'Sunblade'")

class CorpusExtractionResponse(BaseModel):
    triplets: list[FictionalTriplet]

args_dict = dict()
args_dict['text'] = "Marushar is a desolate desert continent around the southern pole of the world, beyond Shovzogr. Unknown to the rest of the world, the deadly group mercenaries called the Hatyaars are based in Marushar, and the endurance they have developed by a surviving there has allowed them to be able to cross Shovzogr at will. Marushar is also the birthplace of Mahamun, although nobody knows it."
args_dict['topics'] = "\n".join([f"{idx+1}. {topic}" for idx, topic in enumerate(["Hatyaars (faction)"])])
args_dict['definitions'] = ""
#

#"Shovzogr is the dead sea that lies to the south of the Archipelago belt. It is so named because there are no currents coming from this sea, and the ones going south die down eventually. No ships can sail south on the Shovzogr because the currents die out, leaving the ship stranded in the still water, away from land. If ships try to return, they are opposed by the currents, being pushed back to the still water. Criminals with the death penalty are shipped into Shovzogr alive in a small boat, where they will die far away from the Archipelago."

#

extraction_prompt = prompts.get_text_decomposition_prompt(args_dict,[])

extraction_prompt[0]['content'] = "You are an advanced triplet extraction engine for fictional lore. Your task is to read the provided text chunk and extract a list of (subject, relation, object) triplets according to the given schema. You must respond ONLY with information related to the given topic."
'''
"[SYSTEM ROLES AND GUARDRAILS]\n 
1. You are an advanced text simplification engine specialized for fictional lore.\n
2. Your task is to read the provided text chunk and simplify it into small, atomic sentences with exactly one subject, one object and one action word ONLY.\n
3. The subject of each atomic sentence MUST come before the object and the action word.\n
4. The subject of each atomic sentence in your response must ALWAYS be the one of the topics provided by the user. You must not respond with atomic sentences with any other subject.\n
5. You must cover all information provided in the text chunk in this manner.\n
6. Do not invent facts.\n
7. Do NOT use any conjunctions such as 'and','with' or 'also'."
8. Do NOT include any conversational preamble or text."
'''
#extraction_prompt[1]['content'] = f"Analyze the following text chunk and simplify it into atomic sentences related ONLY to the following topics.\n\n<topics>\n{topics}\n</topics>\n\n<text_chunk>\n{args_dict['chunk']}\n</text_chunk>\n"

print(extraction_prompt[0]['content'])
print(extraction_prompt[1]['content'])

response = ollama.chat(
    model='llama3.1:8b',
    messages=extraction_prompt,
    options={
        'temperature': 0.0,      # Controls randomness (0.0 = deterministic, 1.0 = creative)
        'num_predict': 512       # Equivalent to max tokens (maximum tokens to generate)
    },
    format = CorpusExtractionResponse.model_json_schema()
)

triplets = CorpusExtractionResponse.model_validate_json(response.message.content)

print(triplets)