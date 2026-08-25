from pydantic import BaseModel, Field, create_model
from typing import Literal

class ReasonedResponse(BaseModel):
    scratchpad: str  # The step-by-step chain of thought
    final_answer: str # The clean answer for the user

class AssumptionsList(BaseModel):
    assumptions: list[str] = Field(description='All the assumptions made by the proposition based on the context provided.')  # The step-by-step chain of thought

class SceneSummary(BaseModel):
    #summary:   str = Field(description='A concise summary of the given text, retaining the important plot points and emotional/thematic shifts. Only use information explicitly mentioned in the text.')
    character: list[str] = Field(description='The names of the sentient characters involved in the scene as a list of strings. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named character is present, output "None".')
    location:  list[str] = Field(description="The name of the location or environment in which the scene is set as a list of strings. STRICT RULE: Only return ONE location name.") 
    event:     list[str] = Field(description="The single most important event happening in the scene as a list of strings. STRICT RULE : Only specify ONE event. The event must involve the characters.") 
    artifacts:  list[str] = Field(description='Fictional artifacts present in the scene as a list of strings. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named artifact is present, output "None".')
    factions:   list[str] = Field(description='The names of factions and organizations mentioned in the scene as a list of strings. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named faction is present, output "None".')

def create_dynamic_choice_model(allowed_options: list[str]):
    # Convert list to a tuple for Literal unpacking
    options_tuple = tuple(allowed_options)
    
    # Use Pydantic's create_model helper to properly annotate and register the field
    DynamicModel = create_model(
        "DynamicChoiceModel",
        selected_option=(Literal[options_tuple], ...) # Tuple structure: (Type, Default/Ellipsis for required)
    )
    return DynamicModel

def create_multi_mcq_model(question_options_dict: dict[str, list[str]]):
    """
    question_options_dict example:
    {
        "who_is_protagonist": ["Alvar", "Lyra", "Unknown"],
        "primary_location": ["Citadel", "Eldoria", "The Wilds"]
    }
    """
    fields = {}
    for field_name, options in question_options_dict.items():
        # Create a Literal constraint for each unique question field
        fields[field_name] = (Literal[tuple(options)], ...)
        
    return create_model("DynamicMultiMCQ", **fields)