import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (AutoTokenizer, BitsAndBytesConfig, LlamaConfig,
                        LlamaForCausalLM) # Keep standard LlamaForCausalLM for the "None" case

# Import the custom model classes from your project structure
# IMPORTANT: Ensure these files (modeling_llamagear.py, modeling_llama_kivi.py)
# are accessible in your Python path when you run the script.
try:
    from modeling_llamagear import LlamaForCausalLM_GEARKIVI
    from modeling_llama_kivi import LlamaForCausalLM_KIVI
except ImportError:
    print("WARNING: Could not import custom model classes (LlamaForCausalLM_GEARKIVI, LlamaForCausalLM_KIVI).")
    print("Ensure modeling_llamagear.py and modeling_llama_kivi.py are in the Python path.")
    # Define dummy classes to allow the script to load if files are missing,
    # but it will likely fail later if custom models are selected via args.model
    class LlamaForCausalLM_GEARKIVI:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("LlamaForCausalLM_GEARKIVI could not be imported.")
    class LlamaForCausalLM_KIVI:
         @staticmethod
         def from_pretrained(*args, **kwargs):
            raise ImportError("LlamaForCausalLM_KIVI could not be imported.")


# Added from evaluation_gsm8k.py for accuracy calculation
def evaluate_pred_answer(pred_str, ans_str):
    """
    Extracts the last numerical value from prediction and answer strings
    and checks if they are equal.
    """
    pattern = r"[-+]?\d*\.?\d+" # Updated pattern to handle potential negative numbers
    # Remove commas for thousands separators
    pred_str = str(pred_str).replace(",", "")
    ans_str = str(ans_str).replace(",", "")

    # Extract final numerical answer from the prediction
    pred_list = re.findall(pattern, pred_str)
    if pred_list:
      try:
        pred = float(pred_list[-1])
      except ValueError:
          pred = None # Handle cases where regex finds something not convertible to float
    else:
        pred = None

    # Extract final numerical answer from the ground truth
    gold_list = re.findall(pattern, ans_str)
    if gold_list:
      try:
        # The actual answer is often formatted like "#### <answer>"
        # We take the last number found.
        gold = float(gold_list[-1])
      except ValueError:
        gold = None # Should ideally not happen for GSM8k gold answers
    else:
        gold = None # Should ideally not happen for GSM8k gold answers

    # Check for equality if both numbers were extracted successfully
    if pred is not None and gold is not None:
        is_pred_true = abs(pred - gold) < 1e-6 # Use tolerance for float comparison
    else:
        is_pred_true = False

    return is_pred_true, pred, gold


# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Model on GSM8K Task")
    # Arguments from original test.py
    parser.add_argument("--model", type=str, default="None", help="Model type identifier (e.g., 'gearl', 'KIVI', 'None' for standard Llama-2). Determines loading logic.")
    parser.add_argument("--model_base_path", type=str, default="meta-llama/Llama-2-7b-hf", help="Base Hugging Face model path (e.g., 'meta-llama/Llama-2-7b-hf').")
    # Arguments for GSM8k evaluation (merged from evaluation_gsm8k.py)
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for generation.")
    parser.add_argument("--prompt_file", type=str, default=None, help="Path to the file containing few-shot prompts. If None, uses zero-shot.")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token for gated models.")
    parser.add_argument("--max_new_tokens", type=int, default=256, help="Maximum number of new tokens to generate.")
    parser.add_argument("--model_max_length", type=int, default=2048, help="Maximum context length for the model and tokenizer.")
    parser.add_argument("--output_dir", type=str, default="gsm8k_results", help="Directory to save results.")
    parser.add_argument("--do_sample", action="store_true", default=False, help="Whether to use sampling; otherwise greedy decoding.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Temperature for sampling.")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k for sampling.") # Default from evaluation_gsm8k.py
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p (nucleus) sampling.") # Default from evaluation_gsm8k.py
    # Arguments for KIVI/GEAR config (from original test.py - adjust defaults if needed)
    parser.add_argument("--k_bits", type=int, default=2, help="K bits for KIVI/GEAR KV cache.")
    parser.add_argument("--v_bits", type=int, default=2, help="V bits for KIVI/GEAR KV cache.")
    parser.add_argument("--group_size", type=int, default=64, help="Group size for KIVI/GEAR.")
    parser.add_argument("--residual_length", type=int, default=64, help="Residual length for KIVI/GEAR.")
    parser.add_argument("--gear_compress_method", type=str, default="gearlKIVI", help="Compress method for GEAR ('gearlKIVI', 'gearsKIVI').")
    parser.add_argument("--gear_quantize_bit", type=int, default=2, help="Quantize bit for GEAR.")
    parser.add_argument("--gear_rank", type=int, default=2, help="Prefill rank K for GEAR.")
    parser.add_argument("--gear_rankv", type=int, default=2, help="Prefill rank V for GEAR.")
    parser.add_argument("--gear_loop", type=int, default=3, help="Loop parameter for GEAR.")


    args = parser.parse_args()

    # --- Setup Output Dir and Logging ---
    output_dir = Path(args.output_dir) / f"{args.model_base_path.split('/')[-1]}_{args.model}_gsm8k"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "evaluation_log.txt"
    results_file = output_dir / "gsm8k_results.json"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info(f"Starting GSM8K evaluation with arguments: {args}")

    # --- Prepare Model Configurations (Original Logic) ---
    config = LlamaConfig.from_pretrained(args.model_base_path, token=args.hf_token)
    # KIVI specific config adjustments
    config.k_bits = args.k_bits
    config.v_bits = args.v_bits
    config.group_size = args.group_size
    config.residual_length = args.residual_length

    # GEAR specific compress_config
    compress_config = {}
    compress_config["compress_method"] = args.gear_compress_method
    compress_config["group_size"] = args.group_size # Note: reused from KIVI args
    compress_config["residual"] = args.residual_length # Note: reused from KIVI args
    compress_config["quantize_bit"] = args.gear_quantize_bit
    compress_config["rank"] = args.gear_rank
    compress_config["rankv"] = args.gear_rankv
    compress_config["loop"] = args.gear_loop
    # stream_list = [torch.cuda.Stream(), torch.cuda.Stream()] # If needed by GEAR
    # compress_config["stream_list"] = stream_list              # If needed by GEAR
    
    quantization_config = BitsAndBytesConfig(load_in_8bit=True) # Optional: If needed
    # compute_dtype = getattr(torch, "bfloat16", torch.float16) # Fallback to float16 if bfloat16 not available
    # logging.info(f"Using compute dtype: {compute_dtype}")
    
    # quantization_config = BitsAndBytesConfig(
    #     load_in_4bit=True,                     # Enable 4-bit quantization
    #     bnb_4bit_quant_type="nf4",             # Use NF4 quantization type
    #     bnb_4bit_compute_dtype=compute_dtype,  # Set compute dtype (bf16 or fp16)
    #     bnb_4bit_use_double_quant=True,        # Enable double quantization
    # )
    # logging.info("Using 4-bit NF4 quantization with double quantization.")

    # --- Load Model Based on args.model (Original Logic) ---
    logging.info(f"Loading model: {args.model_base_path} with type: {args.model}")
    model = None
    if "gearl" in args.model:
        logging.info("Loading LlamaForCausalLM_GEARKIVI model with GEAR config.")
        model = LlamaForCausalLM_GEARKIVI.from_pretrained(
            args.model_base_path,
            config=config, # Pass the potentially modified KIVI config
            compress_config=compress_config,
            quantization_config = quantization_config, # Optional
            device_map="auto", # Changed from "cuda:0" for flexibility
            torch_dtype=torch.float16, # Use float16
            token=args.hf_token
        )
    elif "KIVI" in args.model:
        logging.info("Loading LlamaForCausalLM_KIVI model with KIVI config.")
        model = LlamaForCausalLM_KIVI.from_pretrained(
            args.model_base_path,
            config=config, # Pass the modified KIVI config
            # compress_config=compress_config, # KIVI doesn't take compress_config directly here
            quantization_config = quantization_config, # Optional
            device_map="auto", # Changed from "cuda:0" for flexibility
            torch_dtype=torch.float16, # Use float16
            token=args.hf_token
        )
    elif "None" in args.model or model is None: # Default to standard Llama if 'None' or import failed
        if model is None and ( "gearl" in args.model or "KIVI" in args.model ):
             logging.warning("Failed to load custom model class, falling back to standard LlamaForCausalLM.")
        logging.info("Loading standard LlamaForCausalLM model.")
        model = LlamaForCausalLM.from_pretrained(
            args.model_base_path,
            device_map="auto", # Changed from "cuda:0"
            torch_dtype=torch.float16, # Use float16
            token=args.hf_token,
            quantization_config=quantization_config, # Optional
        )
    else:
        raise ValueError(f"Unknown model type specified: {args.model}. Use 'gearl', 'KIVI', or 'None'.")

    # model = model.half() # Not needed if torch_dtype=torch.float16 is used

    # --- Load Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_base_path,
        model_max_length=args.model_max_length,
        padding_side="left",  # Important for batch generation
        use_fast=False,
        token=args.hf_token,
        trust_remote_code=True # <-- Added back based on original test.py
    )
    # Set pad token if it's not already set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logging.info("Model and tokenizer loaded.")

    # --- Load GSM8K Dataset ---
    logging.info("Loading GSM8K dataset...")
    # Use verification_mode='no_checks' for potentially faster loading if dataset is trusted
    eval_dataset = load_dataset("gsm8k", "main", split="test", verification_mode='no_checks')
    dataloader = DataLoader(eval_dataset, batch_size=args.batch_size)
    logging.info(f"Loaded {len(eval_dataset)} test samples.")

    # --- Prepare Prompt ---
    if args.prompt_file:
        logging.info(f"Loading prompt template from: {args.prompt_file}")
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_template = f.read()
                # print(prompt_template)
        except FileNotFoundError:
            logging.error(f"Prompt file not found: {args.prompt_file}. Exiting.")
            sys.exit(1)
        prompt_prefix = prompt_template + "\nQuestion: "
        prompt_suffix = "\nAnswer:" # Guide the model
    else:
        logging.info("Using zero-shot prompt.")
        prompt_prefix = "Question: "
        prompt_suffix = "\nAnswer:"

    # --- Evaluation Loop ---
    all_samples_results = []
    total_correct = 0
    total_evaluated = 0
    start_time = time.time()

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating GSM8K"):
            questions = batch["question"]
            answers = batch["answer"]

            # Format prompts for the batch
            prompts = [prompt_prefix + q + prompt_suffix for q in questions]

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding="longest",      # Pad to the longest sequence in the batch
                truncation=True,        # Truncate if longer than model_max_length
                max_length=args.model_max_length - args.max_new_tokens # Reserve space for generation
            ).to(model.device) # Move inputs to the same device as the model

            # Configure generation parameters
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            if args.do_sample:
                generate_kwargs["do_sample"] = True
                generate_kwargs["temperature"] = args.temperature
                generate_kwargs["top_k"] = args.top_k
                generate_kwargs["top_p"] = args.top_p
            else: # Greedy decoding
                generate_kwargs["do_sample"] = False

            # Generate responses
            # Ensure generate is called with inputs compatible with the specific model class
            outputs = model.generate(**inputs, **generate_kwargs)
            
            # Decode generated tokens, skipping the prompt part
            input_token_len = inputs.input_ids.shape[1]
            # Handle potential variations in output format if custom models differ
            if hasattr(outputs, 'sequences'): # Standard HF output format
                 output_sequences = outputs.sequences
            else: # Assume outputs *are* the sequences if no 'sequences' attribute
                 output_sequences = outputs

            generations_raw = tokenizer.batch_decode(
                output_sequences[:, input_token_len:],
                skip_special_tokens=True
            )
            print(generations_raw)
          
            # Evaluate each sample in the batch
            for i in range(len(questions)):
                question = questions[i]
                generation = generations_raw[i]
                answer = answers[i]

                is_correct, pred_val, gold_val = evaluate_pred_answer(generation, answer)

                if is_correct:
                    total_correct += 1
                total_evaluated += 1

                all_samples_results.append({
                    "question": question,
                    "prompt": prompts[i],
                    "generation": generation,
                    "answer": answer,
                    "predicted_value": pred_val,
                    "gold_value": gold_val,
                    "is_correct": is_correct
                })

            # Log progress periodically
            if total_evaluated % (args.batch_size * 10) == 0: # Log every 10 batches
                current_accuracy = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
                logging.info(f"Processed {total_evaluated}/{len(eval_dataset)} samples. Current Accuracy: {current_accuracy:.2f}%")


    end_time = time.time()
    # --- Calculate Final Metrics ---
    final_accuracy = (total_correct / total_evaluated) * 100 if total_evaluated > 0 else 0
    evaluation_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device=model.device) / (1024**3) # GB

    logging.info("Evaluation finished.")
    logging.info(f"Total samples evaluated: {total_evaluated}")
    logging.info(f"Total correct: {total_correct}")
    logging.info(f"Final Accuracy: {final_accuracy:.4f}%")
    logging.info(f"Total time: {evaluation_time:.2f} seconds")
    logging.info(f"Peak GPU memory usage: {peak_memory:.2f} GB")

    # --- Save Results ---
    results_data = {
        "args": vars(args),
        "metrics": {
            "accuracy": final_accuracy,
            "total_samples": total_evaluated,
            "total_correct": total_correct,
            "evaluation_time_seconds": evaluation_time,
            "peak_memory_gb": peak_memory,
        },
        "samples": all_samples_results
    }

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)

    logging.info(f"Results saved to: {results_file}")
    logging.info(f"Detailed log saved to: {log_file}")
