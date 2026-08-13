from transformers import AutoModelForCausalLM, AutoTokenizer
from mlx_lm import convert
import json
import torch

with open('config.json', "r", encoding="utf-8") as f:
    config = json.load(f)

model_id = "meta-llama/Llama-3.2-3B-Instruct"
quant_path = config['model_dir']+f'/llama-32-3b-instruct-quantized-4b'

model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto",local_files_only=False)
tokenizer = AutoTokenizer.from_pretrained(model_id,local_files_only=False)

# Run the conversion and quantization
convert(
    model_id,
    quantize=True,
    q_bits=4,              # Choose your target bit-width (e.g., 4 or 8)
    mlx_path=quant_path
)

'''
# Load and quantize
model = AutoAWQForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_id)

quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}
model.quantize(tokenizer, quant_config=quant_config)

# Save quantized model to disk
model.save_quantized(quant_path)
tokenizer.save_pretrained(quant_path)
'''