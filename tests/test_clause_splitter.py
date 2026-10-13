from utils.semantic import SemanticTools
import os
import json
from pathlib import Path
import re

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

sem = SemanticTools(config)


def get_clauses(text,should_be):

    clauses = sem.advanced_split_and_repair(text)
    print(f"\nadvanced_split_and_repair : input = '{text}'\nreturns   {clauses}\nshould be {should_be}")
    return clauses

texts = [
    "Ryland Grace and Eva Stratt are dissimilar people who work in space.",
    "Rocky was sent to Tau Ceti where he discovered the secrets of Astrophage ",
    "Alice bought apples and oranges",
    "Mike and John are friends who meet often.",
    "Mike is a fireman, but then he was laid off.",
    "Mike is a fireman and is responsible for saving several people.",
    "Why was Mike laid off and sent to jail?",
    "How are Mike being a fireman and him being laid off related?",
    "What connects Mike being a fireman and him being laid off?",
    "How would you describe Mike and Josh?",
    "Mike is more friendly with Josh and likes to eat pasta.",
    "Mike and Josh are friends, they hang out together and sometimes hit the gym."
]


should_be = [
    f"{['Ryland Grace Eva Stratt are dissimilar people.', 'Ryland Grace and Eva Stratt work in space.']}",
    f"{['Rocky was sent to Tau Ceti.', 'where Rocky discovered the secrets of Astrophage.']}",
    f"{['Alice bought apples and oranges.']}",
    f"{['Mike John are friends.', 'Mike and John meet often.']}",
    f"{['Mike is a fireman','then Mike was laid off.']}",
    f"{['Mike is a fireman.', 'Mike is responsible for saving several people.']}",
    f"{['Why was Mike laid off?', 'Mike sent to jail?']}",
    f"{['How are Mike being a fireman?","Mike being laid off related?']}",
    f"{['What connects Mike being a fireman?','Mike being laid off?']}",
    f"{['How would you describe Mike and Josh?']}",
    f"{['Mike is more friendly with Josh.', 'Mike likes to eat pasta.']}",
    f"{['Mike Josh are friends' , 'Mike and Josh hang out together.', 'Mike and Josh they sometimes hit the gym.']}"

]

#og_text = "The Blip-A is the ship used by the Eridians to travel to Tau Ceti in search of a cure for Astrophage. The Blip-A is constructed mostly of Xenonite and uses Astrophage as fuel. The Blip-A is 139 meters long. After the Hail Mary spaceship enters the Tau Ceti solar system, the Blip-A heads over to investigate. The Blip-A's name comes from what the Hail Mary's radar labels the first object it encounters. Rocky is the only surviving crew member of the Blip-A, but at one time there were 23 Eridians aboard the ship. However, as their bodies were unaccustomed to the intense radiation of space, every one except for Rocky eventually succumbed to some form of radiation sickness."
#og_text = "As an introduction to himself, Ryland Grace, PhD is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky. Ryland was once a brilliant biologist with wild theories that he hoped to prove true. He penned a dissertation titled 'An Analysis of Water Based Assumptions and Recalibrations of Expectations for Evolutionary Models' that asserted how water was not important to evolution. His paper labelled him an outcast in his field, but he was fired when he called the leading scholar in his field a 'staggering waste of carbon' at a UNESCO conference in Denmark. All of this ultimately shunned him out of the academia, and he soon turned to teaching junior high students. Grace is a tall, Caucasian male with an athletic build, sandy blonde hair, and blue eyes. Dr. Ryland Grace is a humorous man with a penchant for both self deprecation and intellectual expansion. He is great with children, possibly because of his own immaturity. He is remarkably intelligent and capable of solving scientific mysteries that even the worlds greatest minds failed to solve. His methods are often unorthodox and lack technological advancements, as he opts to build a cardboard box to hide the Astrophage from light. Grace is fascinated by the unknown and the universe. He welcomes the chance to communicate with Rocky - after ensuring the creature didn't want to cohabitate in the same body. Prior to his mission, Grace lacked bravery. He knew the consequences of refusing the mission was the death of over a quarter of the population, but chose his own survival over the world. Stratt, however, knew he was their only option. She had him captured, drugged, and put into a coma against his will. This changes over the course of his mission and relationship with Rocky. Rocky is an Eridian engineer from the planet Erid in the 40 Eridani solar system. He was sent to Tau Ceti with twenty-two other Eridian scientists/crewmates to save their star and planet from Astrophage. As an Eridian, he has no facial features and no sense of sight, instead using hearing and echolocation as his primary sense. He communicates using musical tones and chords. Rocky's true name in the Eridian language is a series of musical chords, but Grace calls him 'Rocky' for his rock-like appearance."
#texts = og_text.split('.')

for idx,text in enumerate(texts):

    try:
        clauses = get_clauses(text,should_be[idx])
    except:
        clauses = get_clauses(text,"")
