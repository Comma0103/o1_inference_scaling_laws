import os
import sys
import json
import typing
import time
import logging
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from IPython import embed
import numpy as np
import concurrent.futures
import statistics
from PIL import Image
import re

from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph
from call_gpt import Openai, API_INFOS

model_name_map = {
    "gpt-4o": "OpenAI-GPT-4o",
    "gpt-4o-mini": "OpenAI-GPT-4o-mini"
}

# ================ config ====================
# O1_MODEL = "o1-mini"
O1_MODEL = "gpt-4o-mini"
# OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
OPENAI_CLIENT = Openai(apis=API_INFOS[model_name_map[O1_MODEL]])
PROMPT = """You are a math problem solver. I will give you a problem from the American Invitational Mathematics Examination (AIME). At the end, provide the final answer as a single integer.

Important: You should try your best to use various numbers of total tokens in your reasoning steps.
If you feel like you are finished too early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.

Here's the problem:

{problem}

Think step by step to solve this problem, use various numbers of total tokens in your reasoning, and provide the final answer with "the answer is (X)" where X is a single integer.
"""

TEMPERATURE = 0.8
N_SAMPLE = 200
N_BUCKET = 10
EXTRACT_MODEL = "gpt-4o-mini"
OPENAI_EXTRACT_CLIENT = Openai(apis=API_INFOS[model_name_map[EXTRACT_MODEL]])

MAX_WORKERS = 1
MAX_WORKERS_SINGLE = 1
# ================ config ====================

SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/AIME/sampling/{time_str}'
run_output_dir = '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gpt-4o-mini/AIME/sampling/12-13_06-12_copy'
os.makedirs(run_output_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

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

def get_response(example: dict, cache: dict, idx: int = 0) -> dict:
    if int(example['id']) not in cache and str(example['id']) not in cache:
        cache[example['id']] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'responses': {}}
    elif str(idx) in cache[str(example['id'])]['responses']:
        if idx == N_SAMPLE - 1:
            logging.info(f"Cache hit for problem: {example['problem'][:50]}. idx: {idx}.")
        return cache[str(example['id'])]['responses'][str(idx)]
    
    formatted_prompt = PROMPT.format(problem=example['problem'])
    logging.debug(f"Requesting response for problem starting with: {example['problem'][:50]} running {idx} of {N_SAMPLE} times.")
    response = OPENAI_CLIENT.call(content=formatted_prompt, max_tokens=None, temperature=TEMPERATURE, return_completion=True)
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    cache[example['id']]['responses'][idx] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {example['problem'][:50]}.")
    return result

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
    response['answer_pred'] = answer_pred
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

def generate_single_response(example: dict, cache: dict, idx: int) -> tuple[int, int]:
    response = get_response(example, cache, idx=idx)
    answer = extract_answer(response, cache)
    if answer is None:
        logging.info(f"\nAnswer is None for problem: {example['problem']}, idx: {idx}, token used: {response['tokens']}.\n")
        # answer = 0
    return answer, response['tokens']

def generate_sampled_responses(example: dict, cache: dict) -> list[tuple[int, int]]:
    responses = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_SINGLE) as executor:
        futures = [executor.submit(generate_single_response, example, cache, idx) for idx in range(N_SAMPLE)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
                responses.append((answer, tokens))
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")
    
    logging.info(f"\n\nObtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {responses}.\n\n")
    save_cache(cache, RESPONSE_CACHE_FILENAME)

    return responses

def calculate_bucket_accuracy(dataset: list[dict], cache: dict):

    # Gather all token counts from sampled responses
    all_token_counts = []
    logging.info(f"Sampling responses {N_SAMPLE} times for each problem in {len(dataset)} problems.\n\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_sampled_responses, example, cache) for example in dataset]
        for future in concurrent.futures.as_completed(futures):
            sampled_responses = future.result()
            all_token_counts.extend([tokens for _, tokens in sampled_responses])
    save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Calculate bucket boundaries
    logging.info(f"Calculating bucket boundaries.")
    bucket_boundaries = [round(e) for e in np.percentile(all_token_counts, np.linspace(0, 100, N_BUCKET + 1))]
    logging.info(f"Bucket boundaries: {bucket_boundaries}\n\n")

    # Assign responses to buckets and calculate accuracy
    results_by_bucket = {i: [] for i in range(1, N_BUCKET + 1)}
    logging.info(f"Assigning responses to buckets and calculating accuracy.")
    num_missing_in_bucket = {i: 0 for i in range(1, N_BUCKET + 1)}
    for example in tqdm(dataset, ncols=75):
        sampled_responses = generate_sampled_responses(example, cache)
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
        bucket_accuracies[bucket] = {
            'boundary': (bucket_boundaries[bucket-1], bucket_boundaries[bucket]), 
            'accuracy': accuracy, 
            'num_missing': num_missing_in_bucket[bucket]
        }
        logging.info(f"Bucket {bucket} ({bucket_boundaries[bucket-1]} - {bucket_boundaries[bucket]}): Accuracy {accuracy}")

    return bucket_accuracies

# Main processing
def main():
    cache = get_or_create_cache(RESPONSE_CACHE_FILENAME)
    dataset = load_2024_dataset()
    bucket_accuracies = calculate_bucket_accuracy(dataset, cache)

    # Save final results
    result_file = os.path.join(run_output_dir, "bucket_accuracies.json")
    with open(result_file, 'w') as f:
        json.dump(bucket_accuracies, f, indent=2)

    logging.info(f"\n\nFinal bucket accuracies saved to {result_file}\n\n")
    save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Plot token count vs. accuracy curve
    try:
        logging.info("Generating accuracy plot...")
        boundaries = [bucket_accuracies[b]['boundary'] for b in bucket_accuracies]
        accuracies = [bucket_accuracies[b]['accuracy'] for b in bucket_accuracies]
        lower_bounds = [boundary[0] for boundary in boundaries]

        # Plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 6))
        plt.plot(lower_bounds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        plt.xscale('log', base=2)
        plt.xlabel("Token Count (Lower Boundary)")
        plt.ylabel("Accuracy")
        plt.title(f"Token Count vs. Accuracy for {O1_MODEL}")
        plt.grid(True)
        plt.legend()

        plot_file = os.path.join(run_output_dir, f"accuracy_plot.png")
        plt.savefig(plot_file)
        plt.close()

        logging.info(f"Accuracy plot saved to {plot_file}\n\n")
    except Exception as e:
        logging.error(f"Error generating plot: {str(e)}\n\n")

if __name__ == "__main__":
    main()
