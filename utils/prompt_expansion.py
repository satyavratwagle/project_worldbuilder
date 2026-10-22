from pydantic import BaseModel, Field
from openai import OpenAI

# 1. Define your strict output schema using Pydantic
class ExpandedPredicate(BaseModel):
    raw_phrase: str = Field(description="The original raw phrase extracted from the lore text.")
    canonical_suggestion: str = Field(description="The predicted uppercase canonical predicate (e.g., WARRING_WITH).")
    expanded_definition: str = Field(description="A single, precise definition sentence designed for embedding models, capturing the core action, tone, and direction.")

client = OpenAI()

def expand_predicate_with_gpt4o_mini(raw_phrase: str) -> ExpandedPredicate:
    """
    Uses gpt-4o-mini with native Structured Outputs to expand a raw phrase.
    """
    system_prompt = (
        "You are a semantic expansion engine for a fictional worldbuilding relationship registry. "
        "Take the user's raw relationship phrase and expand it into a rich, definitive sentence "
        "that prevents semantic overlap with antonyms when embedded."
    )

    # OpenAI client handles structured outputs via client.beta.chat.completions.parse
    response = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Raw phrase to expand: '{raw_phrase}'"}
        ],
        response_format=ExpandedPredicate, # Enforces strict schema compliance
    )

    # Returns a fully validated Pydantic instance automatically
    return response.choices[0].message.parsed

# ==========================================
# Example Usage
# ==========================================
if __name__ == "__main__":
    raw_input = "backstabbed"
    
    result = expand_predicate_with_gpt4o_mini(raw_input)
    
    print(f"Raw Phrase:           {result.raw_phrase}")
    print(f"Canonical Suggestion: {result.canonical_suggestion}")
    print(f"Expanded Definition:  {result.expanded_definition}")