from transformers import AutoModelForCausalLM, AutoTokenizer #4.53.2
from mlx_lm import convert
import json
import torch
import mlx_lm
import outlines
from gliner import GLiNER
import json

with open('config.json', "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    config = json.load(f)

extraction_model_id = "urchade/gliner_small-v2.1"
extraction_model = GLiNER.from_pretrained(extraction_model_id)

# 2. Prepare your custom training data (annotated text examples)
with open(config['gliner_training_data_path'], "r", encoding="utf-8") as f:
    # Load the JSON data into a Python dictionary
    train_data = json.load(f)

for idx in range(len(train_data['data'])):
  train_data['data'][idx]['tokenized_text'] = train_data['data'][idx]['tokenized_text'].split(' ')

train_data = train_data['data']


# 3. Fine-tune the model on your custom dataset
extraction_model.train_model(
    train_dataset=train_data,
    eval_dataset=train_data,
    output_dir=config['gliner_dir'],
    max_steps=1000,
    per_device_train_batch_size=8,
    learning_rate=1e-5,
    dataloader_num_workers=0
)