import os
import sys
import json
import typing
import time
import logging
import threading
from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm
from IPython import embed
import numpy as np
import concurrent.futures
import statistics
from PIL import Image
import re
import copy
import sympy as sp
import matplotlib.pyplot as plt

from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph
from call_gpt import Openai, API_INFOS
from math_utils import verify_math_sample

model_name_map = {
    "gpt-4o": "OpenAI-GPT-4o",
    "gpt-4o-mini": "OpenAI-GPT-4o-mini"
}

# ================ config ====================
# O1_MODEL = "o1-mini"
O1_MODEL = "gpt-4o-mini"
# OPENAI_CLIENT = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
OPENAI_CLIENT = Openai(apis=API_INFOS[model_name_map[O1_MODEL]])
MESSAGES_TEMPLATE = [
    {"role": "system", "content": f"You are a helpful and harmless assistant. You are {O1_MODEL} developed by OpenAI. You should think step-by-step."},
    {"role": "user", "content": None}
]
PROMPT = """You are a math problem solver. I will give you a problem from the MATH500 benchmark. At the end, provide the final answer in box.

Important: You should try your best to use various numbers of total tokens in your reasoning steps.
If you feel like you are finished too early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.

Here's the problem:

{problem}

Think step by step to solve this problem, use various numbers of total tokens in your reasoning, and provide the final answer with "{box}" where X is the answer itself.
"""

TEMPERATURE = 0.8
TOP_P = 0.9
N_PROBLEM = 500
N_SAMPLE = 32
N_BUCKET = 10
N_SAMPLES_PER_PROBLEM = 1
FIX_BUCKET_CDF = True
if FIX_BUCKET_CDF:
    BUCKET_STEP = 512

MAX_WORKERS = 8
MAX_WORKERS_SINGLE = 4
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/MATH500/sampling/{time_str}'
# run_output_dir = '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gpt-4o/MATH500/sampling/12-23_04-36_copy'
run_output_dir = '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gpt-4o-mini/MATH500/sampling/12-23_04-37_copy'
os.makedirs(run_output_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile_tmp.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

cache_lock = threading.Lock()


def load_math500():
    datasets_path ='qq8933/MATH500'
    dataset = load_dataset(datasets_path)
    logging.info(f"Load dataset {datasets_path} complete, size: {len(dataset['test'])}") 
    return dataset['test']
    # items = []
    # for item in dataset['test']:
    #     items.append({
    #         'prompt': item['problem'],
    #         'answer': item['answer']
    #     })
    # print(f"load {len(items)} items")
    # return items

def get_or_create_cache(filename: str) -> dict:
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return {}

# def save_cache(cache, filename):
#     with cache_lock:
#         with open(filename, 'w') as f:
#             json.dump(cache, f)

def save_cache(cache, filename):
    success = False
    while not success:
        try:
            # 读取当前文件中的缓存
            try:
                with open(filename, 'r') as f:
                    cache_tmp = json.load(f)
            except FileNotFoundError:
                cache_tmp = {}

            # 检查当前缓存是否比新缓存更new
            tmp_newer = True
            for example_id, example_info in cache.items():
                if not tmp_newer:
                    break
                if example_id not in cache_tmp:
                    tmp_newer = False
                    break
                for idx, response in example_info['responses'].items():
                    if idx not in cache_tmp[example_id]['responses']:
                        tmp_newer = False
                        break
            if tmp_newer:
                logging.info("The existing cache is newer or equally updated. Skipping write.")
                return

            # 写入新的缓存
            with open(filename, 'w') as f:
                json.dump(cache, f, indent=2)
            logging.info(f"Cache saved to {filename}.")

            success = True
        except RuntimeError as e:
            logging.warning(f"RuntimeError encountered: {e}. Retrying in 1 second...")
            time.sleep(1)
        except Exception as e:
            logging.error(f"Unexpected error during save_cache: {e}")
            raise

def get_response(example: dict, cache: dict, idx: int = 0) -> dict:
    with cache_lock:
        if example['unique_id'] not in cache or not cache[example['unique_id']]:
            cache[example['unique_id']] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'subject': example['subject'], 'level': example['level'], 'responses': {}}
        elif str(idx) in cache[example['unique_id']]['responses']:
            if idx == N_SAMPLE - 1:
                logging.info(f"Cache hit for problem: {example['problem'][:50]}. idx: {idx}.")
            return cache[example['unique_id']]['responses'][str(idx)]
    
    messages = copy.deepcopy(MESSAGES_TEMPLATE)
    messages[1]['content'] = PROMPT.format(problem=example['problem'], box=r"\boxed{X}")
    logging.debug(f"Requesting response for problem starting with: {example['problem'][:50]} running {idx} of {N_SAMPLE} times.")
    response = OPENAI_CLIENT.call(messages=messages, max_tokens=None, temperature=TEMPERATURE, top_p=TOP_P, return_completion=True)
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    cache[example['unique_id']]['responses'][idx] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {example['problem'][:50]}.")
    return result

def generate_single_response(example: dict, cache: dict, idx: int) -> tuple[int, int]:
    response = get_response(example, cache, idx=idx)
    correct, predicted_answer = verify_math_sample(response['content'], example['answer'])
    response['answer_pred'] = predicted_answer
    response['correct'] = correct
    if predicted_answer is None:
        logging.info(f"\nAnswer is None for problem: {example['problem'][:50]}, idx: {idx}, token used: {response['tokens']}.\n")
    return predicted_answer, response['tokens'], correct

def generate_sampled_responses(example: dict, cache: dict, example_idx: int) -> list[tuple[str, int]]:
    responses = []
    local_cache = {example['unique_id']: cache.get(example['unique_id'], {})}

    logging.info(f"Sampling responses for problem {example_idx + 1}/500 starting with: {example['problem'][:50]}.\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_SINGLE) as executor:
        futures = [executor.submit(generate_single_response, example, local_cache, idx) for idx in range(N_SAMPLE)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens, correct = future.result()
                responses.append((answer, tokens, correct))
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")

    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {responses}.\n\n")
    with cache_lock:
        cache.update(local_cache)
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    return responses

def calculate_bucket_accuracy(dataset: list[dict], cache: dict):

    dataset = [example for idx, example in enumerate(dataset) if idx < N_PROBLEM] # for testing

    # Gather all token counts from sampled responses
    all_token_counts = []
    logging.info(f"Sampling responses {N_SAMPLE} times for each problem in {len(dataset)} problems.\n\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_sampled_responses, example, cache, idx) for idx, example in enumerate(dataset)]
        for future in concurrent.futures.as_completed(futures):
            sampled_responses = future.result()
            all_token_counts.extend([tokens for _, tokens, _ in sampled_responses])
    with cache_lock:
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    if not FIX_BUCKET_CDF:
        # Calculate bucket boundaries
        logging.info(f"Calculating bucket boundaries.")
        bucket_boundaries = [round(e) for e in np.percentile(all_token_counts, np.linspace(0, 100, N_BUCKET + 1))]
    else:
        # use fixed bucket boundaries
        logging.info(f"Using fixed bucket boundaries.")
        bucket_boundaries = [0] + [BUCKET_STEP*i for i in range(1, N_BUCKET)] + [max(all_token_counts)]
    logging.info(f"Bucket boundaries: {bucket_boundaries}\n\n")

    # Assign responses to buckets and calculate accuracy
    results_by_bucket = {i: [] for i in range(1, N_BUCKET + 1)}
    logging.info(f"Assigning responses to buckets and calculating accuracy.")
    num_missing_in_bucket = {i: 0 for i in range(1, N_BUCKET + 1)}
    example_idx = 0
    for example in tqdm(dataset, ncols=75):
        sampled_responses = generate_sampled_responses(example, cache, example_idx)
        for idx, boundary in enumerate(bucket_boundaries[:-1]):
            bucket_idx = idx + 1
            bucket_responses = [resp for resp in sampled_responses 
                                if bucket_boundaries[idx] <= resp[1] < bucket_boundaries[bucket_idx]]

            if bucket_responses:
                ## choose one response in the bucket
                # random_response = bucket_responses[np.random.randint(0, len(bucket_responses))]
                # score = 1 if random_response[2] else 0
                # results_by_bucket[bucket_idx].append(score)
                ## choose multiple then average
                resample_count = min(len(bucket_responses), N_SAMPLES_PER_PROBLEM)
                resampled_idx = np.random.choice(len(bucket_responses), size=resample_count, replace=False)
                resampled_responses = [bucket_responses[i] for i in resampled_idx]
                scores = [
                    1 if response[2] else 0
                    for response in resampled_responses
                ]
                average_score = np.mean(scores)
                if not FIX_BUCKET_CDF:
                    results_by_bucket[bucket_idx].append(average_score)
                else:
                    for i in range(1, bucket_idx):
                        results_by_bucket[i].append(0)
                    for i in range(bucket_idx, N_BUCKET + 1):
                        results_by_bucket[i].append(average_score)
            else:
                num_missing_in_bucket[bucket_idx] += 1
        example_idx += 1
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
    dataset = load_math500()
    bucket_accuracies = calculate_bucket_accuracy(dataset, cache)

    # Save final results
    result_file = os.path.join(run_output_dir, f"bucket_accuracies{'_fix_bucket_cdf' if FIX_BUCKET_CDF else ''}_{N_SAMPLES_PER_PROBLEM}.json")
    with open(result_file, 'w') as f:
        json.dump(bucket_accuracies, f, indent=2)
    logging.info(f"\n\nFinal bucket accuracies saved to {result_file}\n\n")
    with cache_lock:
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Plot token count vs. accuracy curve
    try:
        logging.info("Generating accuracy plot...")
        boundaries = [bucket_accuracies[b]['boundary'] for b in bucket_accuracies]
        accuracies = [bucket_accuracies[b]['accuracy'] for b in bucket_accuracies]
        lower_bounds = [boundary[0] for boundary in boundaries]

        # Plot
        plt.figure(figsize=(10, 6))
        plt.plot(lower_bounds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
        # plt.xscale('log', base=2)
        plt.xlabel("Token Count (Lower Boundary)")
        plt.ylabel("Accuracy")
        plt.title(f"Token Count vs. Accuracy for {O1_MODEL}")
        plt.grid(True)
        plt.legend()

        plot_file = os.path.join(run_output_dir, f"accuracy_plot{'_fix_bucket_cdf' if FIX_BUCKET_CDF else ''}_{N_SAMPLES_PER_PROBLEM}.png")
        plt.savefig(plot_file)
        plt.close()

        logging.info(f"Accuracy plot saved to {plot_file}\n\n")
    except Exception as e:
        logging.error(f"Error generating plot: {str(e)}\n\n")

if __name__ == "__main__":
    main()
