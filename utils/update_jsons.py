import os
import json

# Ensure tags field exists in raw json files
store_dir = '/Users/satyawagle/Projects/LangChain/llm_chatbot/data/json_store'
files = os.listdir(store_dir)

def uppercase(text):
    return text[0].upper()+text[1:]

documents = []
print(files)
for filename in files:

    if(filename.split('.')[-1]=='json'):

        with open(store_dir+'/'+filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        data['aliases'] = []
        #data['blurb'] = ''
        data['definition'] = data['blurb']
        del data['blurb']

        with open(f"{store_dir}/{filename}", "w") as file:
            json.dump(data, file, indent=4)