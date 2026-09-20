from pydantic import BaseModel, Field, create_model
from typing import Literal, List
from typing_extensions import Annotated
from pydantic.types import conlist

# Class for extracting relationship types from a prose
SnakeCaseStr = Annotated[str, Field(pattern=r"^[a-z]+(_[a-z]+)*$", description="Lowercase snake_case relationship type")]
class RelationshipType(BaseModel):
    relation_types: conlist(SnakeCaseStr, min_length=1, max_length=20) = Field(description="A unique list of high-value, story-driven relationship types derived from the provided text.")

class ReasonedResponse(BaseModel):
    scratchpad: str  # The step-by-step chain of thought
    answer: str # The clean answer for the user

class Hypothesis(BaseModel):
    deduction: str = Field(description="A derived deduction or inference drawn from the given facts.")
    reasoning: str = Field(description="The logical basis for the deduction based on the given facts.")
    
class HypothesisList(BaseModel):
    deductions: conlist(Hypothesis, min_length=1, max_length=5) = Field(description="A list of extracted deduction-reasoning pairs drawn from the given facts.")

class AssumptionsList(BaseModel):
    assumptions: list[str] = Field(description='All the assumptions made by the proposition based on the context provided.')  # The step-by-step chain of thought

