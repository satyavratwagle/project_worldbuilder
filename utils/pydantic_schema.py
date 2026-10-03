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
    genre: Literal["location", "process", "attribute", "time_period"] = Field(description="The category of information needed to answer the user query.")

# Text Decomposition Schema
class TextTuple(BaseModel):
    atomic_sentence: str = Field(description="An atomic sentence related to one of the user topics.")
    topic: str = Field(description="Subject of 'atomic_sentence'")
    label: Literal["AXIOM","INCIDENT"] = Field(description="Choose AXIOM only if 'atomic_sentence' describes permanent truths, core identities, immutable facts, or long-term properties that rarely or never change (e.g., parent-child, material composition, birthplace). Choose INCIDENT only 'atomic_sentence' describes temporary actions, plot events, transient states, situational possessions, or active conflicts that change or expire (e.g., carrying an item, fighting an enemy, temporary possession).")

class DecomposedText(BaseModel):
    decomposed_text: list[TextTuple]

# Node Summary Schema
# Text Decomposition Schema
class NodeSummary(BaseModel):
    lore:       str = Field(description="A summary of the lore-related information related to the topic.")
    plot:        str = Field(description="A summary of the plot-related information related to the topic.")

label_map = {'AXIOM' : "Lore", 'INCIDENT': "Plot"}

# Plot Reordering Schema
class Reordering(BaseModel):
    thinking: str = Field(description="Step-by-step chain of thought to determine the chronological relationship between the given events.")
    sequential_events: bool = Field(description="'True' if the given events can be split into 'before' and 'after' subsets based on the given story outline.")
    before: Optional[list[str]] = Field(description="A list of sentence IDs corresponding to the events that happened first based on your thinking.")
    after: Optional[list[str]] = Field(description="A list of sentence IDs corresponding to the events that happened later based on your thinking.")

class PlotPhase(BaseModel):
    index : int = Field(ge=1,description="The index of the phase in chronological order.")
    ids : list[str] = Field(description="A list of sentence IDs corresponding to events happening concurrently or overlapping in time.")

class PhaseList(BaseModel):
    phases: list[PlotPhase] = Field(description="A list of CHRONOLOGICALLY ORDERED plot phases.")

class Triplet(BaseModel):
    source: str = Field(description="Normalized canonical name of the source entity in simple_sentence, e.g., 'Prince Elian'")
    target: str = Field(description="Normalized canonical name of the target entity in simple_sentence, e.g., 'Sunblade'")
    relation: str = Field(description="A description of the relation between the source and target in  in simple_sentence.")

class TripletList(BaseModel):
    simplified_text: str
    triplets: list[Triplet]