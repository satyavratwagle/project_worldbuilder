from pydantic import BaseModel, Field

class ReasonedResponse(BaseModel):
    scratchpad: str  # The step-by-step chain of thought
    final_answer: str # The clean answer for the user

class SceneSummary(BaseModel):
    characters: list
    locations: list 
    events: list 
    artifacts: list
    factions: list

