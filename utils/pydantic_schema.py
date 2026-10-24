from pydantic import BaseModel, Field, create_model
from typing import Literal, List, Optional
from typing_extensions import Annotated
from pydantic.types import conlist
import json
from enum import Enum

def create_custom_triplet_schema(topics):

    class Triplet(BaseModel):        
        reasoning: str = Field(description="Identify topics and describe which topic is acting on which topic.")
        simplified_text: str = Field(description="Simple sentences describing the statement being analyzed.")
        source: Literal[tuple(topics)] = Field(description="The subject of the 'simplified_text'.")
        target: Literal[tuple(topics)] = Field(description="The object of the 'simplified_text'.")
        relation: str = Field(description="The relation between the 'source' and 'target'.")

    class TripletList(BaseModel):   
        triplets: list[Triplet]

    return TripletList

class RelationCategory(str, Enum):
    STRUCTURAL = "Structural & Compositional"
    SPATIAL = "Spatial & Topological"
    TEMPORAL = "Temporal & Causal"
    SOCIAL = "Social & Organizational"
    FUNCTIONAL = "Functional & Operational"

class EntityType(str, Enum):
    AGENT = "Agents & Collectives"
    PLACE = "Places & Spaces"
    OBJECT = "Objects & Artifacts"
    CONCEPT = "Concepts & Information"
    EVENT = "Events & Phenomena"

with open('utils/semantic_definitions.json', "r", encoding="utf-8") as f:
    definitions = json.load(f)

entity_categories = [key for key in definitions['entity_types'].keys()]
label_map = {'AXIOM' : "Lore", 'INCIDENT': "Plot"}

relation_categories = [defn['definition'] for key,defn in definitions['relation_types'].items()]

# Class for extracting relationship types from a prose
class EntityDiscovery(BaseModel):
    entity_name: str = Field(description="The name of the entity discovered.")
    reasoning: str = Field(description="A brief, one-sentence analysis of the core nature of the entity and justification of why it belongs to the chosen primary category.")
    entity_type: EntityType = Field(
        description=(
            "The primary entity category. Must be one of: "
            "1. Agents & Collectives (people, groups, institutions); "
            "2. Places & Spaces (locations, regions, structures); "
            "3. Objects & Artifacts (tangible items, equipment, assets); "
            "4. Concepts & Information (abstract ideas, data, lore, laws); "
            "5. Events & Phenomena (time-bound occurrences, battles, processes)."
        )
    )
    entity_subcategory: str = Field(description="A one-word sub-category of the entity, a type of the primary entity category.")

class EntityDiscoveryList(BaseModel):
    entities_discovered: list[EntityDiscovery] = Field(description="A list of discovered entities.")

class RelationshipType(BaseModel):
    reasoning: str = Field(description="A brief, one-sentence analysis of the core nature of the relationship and justification of why it belongs to the chosen primary category.")
    relation_type: RelationCategory = Field(
        description=("The primary relation category. Must be one of: "
                    "1. Structural & Compositional (identity, hierarchies, parts); "
                    "2. Spatial & Topological (geography, containment, borders); "
                    "3. Temporal & Causal (event ordering, causality, triggers); "
                    "4. Social & Organizational (affiliations, power, conflict); "
                    "5. Functional & Operational (actions, tools, dependencies, outputs)."))
    relation_subcategory: str = Field(description="A one-word sub-category of the relation, a type of the primary relation category.")

# 1. Keep Enum values short and clean for database indexing
# 2. Use Field descriptions for schema-level context in OpenAI Structured Outputs
class EntityType(BaseModel):
    reasoning: str = Field(description="A brief, one-sentence analysis of the core nature of the entity and justification of why it belongs to the chosen primary category.")
    entity_type: EntityType = Field(
        description=(
            "The primary entity category. Must be one of: "
            "1. Agents & Collectives (people, groups, institutions); "
            "2. Places & Spaces (locations, regions, structures); "
            "3. Objects & Artifacts (tangible items, equipment, assets); "
            "4. Concepts & Information (abstract ideas, data, lore, laws); "
            "5. Events & Phenomena (time-bound occurrences, battles, processes)."
        )
    )
    entity_subcategory: str = Field(description="A one-word sub-category of the entity, a type of the primary entity category.")

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
    fact: str = Field(description="A fact related to one of the user topics.")
    topic: str = Field(description="Subject of 'fact'")
    label: Literal["AXIOM","INCIDENT"] = Field(description="Choose AXIOM only if 'fact' describes permanent truths, core identities, immutable facts, or long-term properties that rarely or never change (e.g., parent-child, material composition, birthplace). Choose INCIDENT only 'fact' describes temporary actions, plot events, transient states, situational possessions, or active conflicts that change or expire (e.g., carrying an item, fighting an enemy, temporary possession).")

class DecomposedText(BaseModel):
    decomposed_text: list[TextTuple]

# Node Summary Schema
class NodeSummary(BaseModel):
    category: Literal[tuple(entity_categories)] = Field(description="The category of the given topic.")
    summary: str = Field(description="A summary of the of the information related to the specified topic.")

# Lore Cluster Theme Schema

class ClusterDescription(BaseModel):
    summary: str = Field(description="Summary of a description.")
    theme: str = Field(description="Theme of a descriptions.")

class ClusterThemes(BaseModel):
    description_summaries: conlist(ClusterDescription) = Field(description="Summaries and themes of each of the descriptions.")

# Entity Relationship Schema
class RelationshipCategory(BaseModel):
    category: Literal[tuple(relation_categories)] = Field(description="The category of the relationship between head and tail.")

# Context Sufficiency Schema
class MultipleChoiceAnswer(BaseModel):
    question: str = Field(description="A multiple-choice question related to provided that will help you answer the user query")
    options: conlist(str,min_length=3,max_length=3) = Field(description="A list of EXACTLY 3 (three) options that could potentially answer the question correctly.")

class SufficiencyCheck(BaseModel):
    relevant_information: str = Field(description="Parts of the context that are relevant to the user query.")
    information_needed: str = Field(description="A description of the information gaps in the provided context that prevent you from answering the query correctly.")
    is_sufficient: int = Field(ge=0, le=10, description="A measure of whether the provided context contains enough information to answer the user query. 0 if the context contains NO relevant information. 10 if the context contains ALL required information. 5 if the user query can only be partially answered using the provided context.")
    additional_questions: conlist(MultipleChoiceAnswer, min_length=1, max_length=3) = Field(description="A set of multiple-choice questions based on the 'information_needed' field whose responses will provide the additional information needed to answer the question.")

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
    thinking: str = Field(description="The given story outline in chornological order without any preamble.")
    event_id: str = Field(description="The ID of the event that happened first based on the given story outline.")

class Triplet(BaseModel):
    source: str = Field(description="The subject of the statement.")
    target: str = Field(description="The object of the statement")
    relation: str = Field(description="The relation between the 'source' and 'target'.")

class TripletList(BaseModel):
    triplets: list[Triplet]