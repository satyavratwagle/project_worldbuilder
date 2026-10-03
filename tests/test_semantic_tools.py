from utils.semantic import SemanticTools
import json

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

sem = SemanticTools(config)
#sem.load_extraction_model()

og_text = "Eva Stratt has a singular motivation - saving Earth from Astrophage. She is singularly determined, headstrong, intelligent, and willing to make tough decisions to get the desirable outcome. Ryland Grace believes Stratt is mysterious as she doesn't speak often, nor does she interact with the crew outside of a professional setting. She surprises him by singing Sign of the Times at karaoke night, further bonding them. She also has a dry sense of humor as opposed to Grace's wit. When the scientists meant to board the Hail Mary are blown up in a miscalculation, she informs Grace that he will be taking their spots on the shuttle. He refuses, believing he has a choice - which he does not. She ultimately has him chased, bound, drugged, and placed into a medically induced coma against his will so they can place him on the ship. She hoped that one day he would understand the decisions she made."

#"Rocky is roughly the size of a Labrador retriever. His exterior possesses a smooth, rock-like texture. His skin color varies from blackish-brown to brown. He has five limb-like appendages, and is described as 'similar in appearance to a spider.' On each limb, Rocky has three triangular shaped fingers, none appearing to function as a thumb. The top of his carapace has a pentagonal outcropping of the same rock-like material as the rest of his body. Carvings cover most of his appendages, displaying his family crest and status as his ship's engineer, and he also has spots of a green gel or stone-like substance. He lacks a face and uses sonar or echolocation to 'see'. A hole in the bottom of his body opens to act as both an entry for food and as an exit for waste. After risking his life to save Grace and exposing himself to Earth's atmosphere, Rocky's crevices leak mercury."


#"Ryland Grace is a self-deprecating 8th grade science teacher who was shunned from higher education and became the laughing stock of his scientific peers after releasing a paper on why water lacks importance for evolution. Grace was then recruited by Eva Stratt to solve the mystery of Astrophage due to that same scientific paper. His brilliance and knowledge on Astrophage makes him an unwilling member of the Hail Mary space expedition, where he's placed into a medically induced coma for four years. He awakens with no memory of his life, his cohorts are dead, and the only life form he encounters is an alien he calls Rocky. Dr. Ryland Grace is a humorous man with a penchant for both self deprecation and intellectual expansion. He is great with children, possibly because of his own immaturity. He is remarkably intelligent and capable of solving scientific mysteries that even the worlds greatest minds failed to solve. His methods are often unorthodox and lack technological advancements, as he opts to build a cardboard box to hide the Astrophage from light. The Astrophage tries to escape, but it cannot. Grace is fascinated by the unknown and the universe. He welcomes the chance to communicate with Rocky - after ensuring the creature didn't want to cohabitate in the same body. Prior to his mission, Grace lacked bravery. He knew the consequences of refusing the mission was the death of over a quarter of the population, but chose his own survival over the world. Eva Stratt, however, knew he was their only option. Stratt had him captured, drugged, and put into a coma against his will. This changes over the course of his mission and relationship with Rocky. Grace is a tall, Caucasian male with an athletic build, sandy blonde hair, and blue eyes."
e_dict = dict()
e_dict['Doug'] = dict()
e_dict['Doug']['aliases'] = ['Ryland Grace']

e_dict['Eva Stratt'] = dict()
e_dict['Eva Stratt']['aliases'] = []

doc, entity_coreferences = sem.get_alias_clusters(og_text,e_dict)
resolved_text = sem.resolve_coreferences(doc,entity_coreferences)

print(resolved_text)