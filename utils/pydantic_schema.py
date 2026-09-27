from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Optional
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

# Query Evaluation Schema
class QueryEvaluationSchema(BaseModel):
    is_context_sufficient: bool = Field(description="'True' if the provided context is sufficient to completely answer the user query. 'False' otherwise.")
    new_query: Optional[str] = Field(description="If provided context is not sufficient, a new data query for the missing context.")
    answer: Optional[str] = Field(description="If provided context is sufficient, the answer to the user query.")
    genre: Literal["location", "process", "attribute", "time period"] = Field(description="The category of information needed to answer the user query.")

# Text Decomposition Schema
class TextTuple(BaseModel):
    atomic_sentence: str = Field(description="An atomic sentence related to one of the user topics.")
    topic: str = Field(description="Subject of 'atomic_sentence'")
    label: Literal["location", "process", "attribute", "time period"] = Field(description="Label assigned to 'atomic_sentence'")

class DecomposedText(BaseModel):
    decomposed_text: list[TextTuple]

# Node Summary Schema
# Text Decomposition Schema
class NodeSummary(BaseModel):
    location:       str = Field(description="A summary of the location-related descriptions of the topic.")
    process:        str = Field(description="A summary of the process-related descriptions of topic.")
    attribute:      str = Field(description="A summary of the attribute-related descriptions of the topic.")
    time_period:    str = Field(description="A summary of the time period-related descriptions of the topic.")


label_map = dict()
label_map['location'] = 'Location'
label_map['process'] = 'Process'
label_map['attribute'] = 'Attribute'
label_map['time_period'] = 'Time Period'

class Triplet(BaseModel):
    source: str = Field(description="Normalized canonical name of the source entity in simple_sentence, e.g., 'Prince Elian'")
    target: str = Field(description="Normalized canonical name of the target entity in simple_sentence, e.g., 'Sunblade'")
    relation: str = Field(description="A description of the relation between the source and target in  in simple_sentence.")

class TripletList(BaseModel):
    simplified_text: str
    triplets: list[Triplet]