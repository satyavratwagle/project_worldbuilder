from utils.semantic import SemanticTools
import json

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)
sem = SemanticTools(config)

words = ['LOCATED_BEYOND','LOCATED_AT','BASED_IN','ENABLED_TO_CROSS','DEVELOPED_ENDURANCE_IN','BORN_IN']
word1 = sem.nlp("located in")

for word in words:
	new_word = ' '.join(word.lower().split('_'))
	word2 = sem.nlp(new_word)
	print(f"{word} : {word1.similarity(word2)}")