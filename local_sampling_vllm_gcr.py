import os
import sys
import json
import typing
import time
import logging
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from openai import OpenAI
from tqdm import tqdm
from IPython import embed
import numpy as np
import concurrent.futures
import statistics
from PIL import Image
import re

from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph

model_path_map = {
    "QwQ-32B-Preview": "/mnt/qilongma/public_models/QwQ-32B-Preview",
    "Qwen2.5-32B-Instruct": "/mnt/qilongma/public_models/Qwen2.5-32B-Instruct",
}

# ================ config ====================
# O1_MODEL = "o1-mini"
O1_MODEL = "QwQ-32B-Preview"
PROMPT = """You are a math problem solver. I will give you a problem from the American Invitational Mathematics Examination (AIME). At the end, provide the final answer as a single integer.

Important: You should try your best to use various numbers of total tokens in your reasoning steps.
If you feel like you are finished too early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.

Here's the problem:

{problem}

Think step by step to solve this problem, use various numbers of total tokens in your reasoning, and provide the final answer with "the answer is (X)" where X is a single integer.
"""
TEMPERATURE = 0.8
MAX_MODEL_TOKENS = 32768
MAX_NEW_TOKENS = 32768 - 2048
GPU_UTIL = 0.9
N_SAMPLE = 200
N_BUCKET = 10

MAX_WORKERS = 1
# ================ config ====================


SAVE_DIR = f'/mnt/qilongma/inf_scal_law/results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/AIME/sampling/{time_str}'
os.makedirs(run_output_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


def load_model():
    llm = LLM(
        model=model_path_map[O1_MODEL], gpu_memory_utilization=float(GPU_UTIL),
        tensor_parallel_size=torch.cuda.device_count(),
        max_model_len=MAX_MODEL_TOKENS,
        trust_remote_code=True
    )
    logging.info(f"Load model {model_path_map[O1_MODEL]} complete!")
    tokenizer = AutoTokenizer.from_pretrained(model_path_map[O1_MODEL], trust_remote_code=True)
    logging.info(f"Load tokenizer complete!")
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=[
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("Question:")
        ]
    )
    return (llm, sampling_params), tokenizer

def load_2024_dataset() -> list[dict]:
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])
    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
    return dataset

def get_or_create_cache(filename: str) -> dict:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)


def extract_answer(response: dict, cache: dict) -> int:
    if 'answer_pred' in response:
        return response['answer_pred']

    pattern = r"[aA]nswer is \(?(\d+)\)?" # 在文本中找到以 answer is 开头，后面紧跟的一个正整数的字符串
    try:
        match = re.search(pattern, str(response['content']))
    except Exception as e:
        logging.debug(f"Error {str(e)} in extracting answer in response: " + response['content'])
    if match:
        extracted_answer = match.group(1)
    else:
        logging.debug("1st answer extract failed in: \n" + response['content'])
        extracted_answer = extract_again(response['content'])
    
    answer_pred = int(extracted_answer) if extracted_answer else None
    return answer_pred

def extract_again(text):
    match = re.search(r".*[aA]nswer:\s*(\d+)", text) # 最后一个匹配 Answer: 或 answer: 后紧跟的一个正整数的字符串
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b\d+\b(?!.*\b\d+\b)" # 文本中最后一个独立的正整数
    match = re.search(pattern, text, re.DOTALL)
    return match.group(0) if match else None


def batch_inference(llm, sampling_params, inference_batch, cache, example_id):
    start = time.time()
    outputs = llm.generate(inference_batch, sampling_params)
    logging.info("Inference batch, size: " + str(len(inference_batch)) + ", costing time: " + str(time.time() - start))
    results_batch = []
    for idx, output in enumerate(outputs):
        generated_text = output.outputs[0].text
        tokens = len(output.outputs[0].token_ids)
        response = {
            'content': generated_text,
            'tokens': tokens
        }
        cache[example_id]['responses'][idx] = response
        logging.debug(f"Received {response['tokens']} tokens as response, idx: {idx}.")
        answer_pred = extract_answer(response, cache)
        response['answer_pred'] = answer_pred
        if answer_pred is None:
            logging.info(f"\nAnswer is None for idx: {idx}, token used: {response['tokens']}.\n")
        results_batch.append((answer_pred, tokens))
    return results_batch

def generate_sampled_responses(example: dict, model, tokenizer, cache: dict) -> list[tuple[int, int]]:
    if example['id'] not in cache:
        cache[example['id']] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'responses': {}}
    elif len(cache[example['id']]['responses']) >= N_SAMPLE:
        logging.debug(f"Cache hit for problem: {example['problem'][:50]}.")
        return [(resp['answer_pred'], resp['tokens']) for resp in cache[example['id']]['responses'].values()]

    llm, sampling_params = model
    inference_batch = []

    logging.info("generating prompts for: " + example['problem'][:50] + "\n")
    for idx in tqdm(range(N_SAMPLE), ncols=75):
        formatted_prompt = PROMPT.format(problem=example['problem'])
        inference_batch.append(formatted_prompt)
    
    logging.info("evaluating: " + example['problem'][:50])
    responses = batch_inference(llm, sampling_params, inference_batch, cache, example['id'])
    logging.info("\n")
    
    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {responses}.\n\n")
    save_cache(cache, RESPONSE_CACHE_FILENAME)

    return responses

def calculate_bucket_accuracy(dataset: list[dict], model, tokenizer, cache: dict):

    # Gather all token counts from sampled responses
    all_token_counts = []
    logging.info(f"Sampling responses for {len(dataset)} problems.\n\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_sampled_responses, example, model, tokenizer, cache) for example in dataset]
        for future in concurrent.futures.as_completed(futures):
            sampled_responses = future.result()
            all_token_counts.extend([tokens for _, tokens in sampled_responses])
    save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Calculate bucket boundaries
    logging.info(f"Calculating bucket boundaries.")
    bucket_boundaries = np.percentile(all_token_counts, np.linspace(0, 100, N_BUCKET + 1))
    logging.info(f"Bucket boundaries: {bucket_boundaries}\n\n")

    # Assign responses to buckets and calculate accuracy
    results_by_bucket = {i: [] for i in range(1, N_BUCKET + 1)}
    logging.info(f"Assigning responses to buckets and calculating accuracy.")
    num_missing_in_bucket = {i: 0 for i in range(1, N_BUCKET + 1)}
    for example in tqdm(dataset, ncols=75):
        sampled_responses = generate_sampled_responses(example, model, tokenizer, cache)
        for idx, boundary in enumerate(bucket_boundaries[:-1]):
            bucket_idx = idx + 1
            bucket_responses = [resp for resp in sampled_responses 
                                if bucket_boundaries[idx] <= resp[1] < bucket_boundaries[bucket_idx]]

            if bucket_responses:
                random_response = bucket_responses[np.random.randint(0, len(bucket_responses))]
                score = 1 if random_response[0] is not None and int(example['answer']) == int(random_response[0]) else 0
                results_by_bucket[bucket_idx].append(score)
            else:
                num_missing_in_bucket[bucket_idx] += 1
    logging.info(f"Number of missing responses in each bucket: {num_missing_in_bucket}\n\n")

    # Calculate and log accuracy for each bucket
    bucket_accuracies = {}
    for bucket, scores in results_by_bucket.items():
        accuracy = np.mean(scores) if scores else 0
        bucket_accuracies[bucket] = accuracy
        logging.info(f"Bucket {bucket} ({bucket_boundaries[bucket-1]} - {bucket_boundaries[bucket]}): Accuracy {accuracy}")

    return bucket_accuracies

# Main processing
def main():
    cache = get_or_create_cache(RESPONSE_CACHE_FILENAME)
    model, tokenizer = load_model()
    dataset = load_2024_dataset()
    bucket_accuracies = calculate_bucket_accuracy(dataset, model, tokenizer, cache)

    # Save final results
    result_file = os.path.join(run_output_dir, "bucket_accuracies.json")
    with open(result_file, 'w') as f:
        json.dump(bucket_accuracies, f, indent=2)

    logging.info(f"\n\nFinal bucket accuracies saved to {result_file}")
    save_cache(cache, RESPONSE_CACHE_FILENAME)

if __name__ == "__main__":
    main()
