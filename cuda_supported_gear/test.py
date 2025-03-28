from modeling_llamagear import LlamaForCausalLM_GEARKIVI
from modeling_llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse

# Argument parser
parser = argparse.ArgumentParser(description="Evaluate GSM8K Dataset")
parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
parser.add_argument("--compress_method", type=str, default="gearlKIVI", help="Type of compression method.")
args = parser.parse_args()

# Model and tokenization configuration
quantization_config = BitsAndBytesConfig(load_in_8bit=True)
max_token = 1000  # Prefill length
max_generation_length = 1500  # Generate 500 tokens
batch_size = args.batch_size

config = LlamaConfig.from_pretrained(args.model)
config.k_bits = 2  # Current support: 2/4 bit for KV Cache
config.v_bits = 2
config.group_size = 64
config.residual_length = 64  # Number of recent fp16 tokens

# Compression configuration
compress_config = {
    "compress_method": args.compress_method,
    "group_size": 64,
    "residual": 64,
    "quantize_bit": 2,
    "rank": 2,
    "rankv": 2,
    "loop": 3
}

stream_list = [torch.cuda.Stream(), torch.cuda.Stream()]
if "gearl" in args.compress_method:
    model = LlamaForCausalLM_GEARKIVI.from_pretrained(
        args.model,
        config=config,
        quantization_config=quantization_config,
        compress_config=compress_config,
        device_map="cuda:0"
    )
elif "KIVI" in args.compress_method:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        args.model,
        config=config,
        quantization_config=quantization_config,
        device_map="cuda:0"
    )
else:
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        device_map="cuda:0"
    )

print(f"MODEL CONFIG: {model.config}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    args.model,
    model_max_length=max_token,
    max_length=max_token,
    use_fast=False,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# Load GSM8K dataset
dataset = load_dataset("gsm8k", "main", split="test")
questions = dataset["question"]
answers = dataset["answer"]

def evaluate_model():
    correct = 0
    total = 0
    
    for i in range(0, len(questions), batch_size):
        batch_questions = questions[i:i+batch_size]
        inputs = tokenizer(batch_questions, return_tensors="pt", padding=True, truncation=True).to("cuda:0")
        
        # Generate responses
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=max_generation_length, use_cache=True)
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Compare predictions to ground truth
        for pred, actual in zip(generated_texts, answers[i:i+batch_size]):
            if pred.strip() == actual.strip():
                correct += 1
            total += 1
    
    accuracy = correct / total * 100
    print(f"Final Accuracy: {accuracy:.2f}%")

# Run evaluation
evaluate_model()
