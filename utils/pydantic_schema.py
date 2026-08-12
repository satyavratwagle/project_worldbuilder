from pydantic import BaseModel, Field

class ReasonedResponse(BaseModel):
    scratchpad: str  # The step-by-step chain of thought
    final_answer: str # The clean answer for the user

class SceneSummary(BaseModel):
    character: list = Field(description='The names of the sentient characters involved in the scene. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named character is present, output "None".')
    location:  list = Field(description="The name of the location or environment in which the scene is set. STRICT RULE: Only return ONE location name.") 
    event:     list = Field(description="The single most important event happening in the scene. STRICT RULE : Only specify ONE event. The event must involve the characters.") 
    artifacts:  list = Field(description='Fictional artifacts present in the scene. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named artifact is present, output "None".')
    factions:   list = Field(description='The names of factions and organizations mentioned in the scene. STRICT RULE: Do NOT extract common nouns, inanimate objects, body parts, or generic items (e.g., never extract "sword", "shadow", "wind", "dog", or "stranger" unless it is a specific proper name). If no named faction is present, output "None".')

