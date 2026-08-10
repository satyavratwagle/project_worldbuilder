from pydantic import BaseModel, Field,field_validator
from typing import List
import outlines

def get_schema_description(list_schema):

	json_schema = list_schema.model_json_schema()
	key = list(json_schema['$defs'].keys())[0]
	properties = json_schema['$defs'][key]['required']

	descriptions = [json_schema['$defs'][key]['properties'][p]['description'] for p in properties]

	return ', '.join(descriptions).lower()


# 1. Define the target JSON structure via Pydantic
class RelationTriple(BaseModel):
    subject: str = Field(description="The source entity, e.g., character, faction, or artifact.")
    relation: str = Field(description="The relational verb or connection, formatted in uppercase.")
    object: str = Field(description="The target entity being interacted with.")

class KnowledgeGraphSchema(BaseModel):
    triples: List[RelationTriple]

class CharacterSchema(BaseModel):
    full_name: str = Field(description="full name")
    occupation: str = Field(description="occupation or profession")
    appearance: str = Field(description="appearance")
    personality: str = Field(description="personality")
    relationships: str = Field(description="relationships with other characters")

class LocationSchema(BaseModel):
    name: str = Field(description="full name")
    geography: str = Field(description="occupation or profession")
    architecture: str = Field(description="appearance")
    demographics: str = Field(description="personality")
    government: str = Field(description="relationships with other characters")
    history: str = Field(description="relationships with other characters")

class RelationshipSchema(BaseModel):
    subject: str = Field(description="The primary character")
    object: str = Field(description="The character with whom the primary character has a relationship")
    relationship: str = Field(description="A brief description of the relationship between the two characters")

class CharacterListSchema(BaseModel):
    information_list: List[CharacterSchema]

class LocationListSchema(BaseModel):
    information_list: List[LocationSchema]
