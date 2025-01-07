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
from call_gemini import Gemini


# ================ config ====================
# O1_MODEL = "o1-mini"
GEMINI_MODEL = "gemini-2.0-flash-thinking-exp-1219"
GEMINI_CLIENT = Gemini(
    model_name=GEMINI_MODEL,
    sys_inst="You are a helpful and harmless assistant. You are Gemini developed by Google. You should think step-by-step."
)
PROMPT = """You are a math problem solver. I will give you a problem from the MATH500 benchmark. At the end, provide the final answer in box.

Important: You should try your best to use various numbers of total tokens in your reasoning steps.
If you feel like you are finished too early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.

Here's the problem:

{problem}

Think step by step to solve this problem, use various numbers of total tokens in your reasoning, and provide the final answer with "{box}" where X is the answer itself.
"""

TEMPERATURE = 0.8
TOP_P = 0.9
N_PROBLEM = 300
N_SAMPLE = 16
N_BUCKET = 10
N_SAMPLES_PER_PROBLEM = 1
FIX_BUCKET_CDF = True
if FIX_BUCKET_CDF:
    BUCKET_STEP = 512

MAX_WORKERS = 2
MAX_WORKERS_SINGLE = 1
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{GEMINI_MODEL}/MATH500/sampling/{time_str}'
run_output_dir = '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gemini-2.0-flash-thinking-exp-1219/MATH500/sampling/12-30_11-20'
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
    
    content = PROMPT.format(problem=example['problem'], box=r"\boxed{X}")
    logging.debug(f"Requesting response for problem starting with: {example['problem'][:50]} running {idx} of {N_SAMPLE} times.")
    response = GEMINI_CLIENT.generate(content=content, max_output_tokens=None, temperature=TEMPERATURE, top_p=TOP_P, return_completion=True)
    result = {
        'content': response.text,
        'tokens': response.usage_metadata.candidates_token_count
    }
    cache[example['unique_id']]['responses'][idx] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {example['problem'][:50]}.")
    return result

# def extract_answer(response: dict, cache: dict) -> str:
#     if 'answer_pred' in response:
#         return response['answer_pred']

#     pattern = r"\\boxed\{(.*?)\}" # r"\[ \\boxed\{(.*?)\} \\" # 提取最后一个 \boxed{...} 中的内容
#     try:
#         match = re.search(pattern, str(response['content']))
#     except Exception as e:
#         logging.debug(f"Error {str(e)} in extracting answer in response: " + response['content'])
#     if match:
#         extracted_answer = match.group(1).strip()
#     else:
#         logging.debug("1st answer extract failed in: \n" + response['content'])
#         # extracted_answer = extract_again(response['content'])
#         extracted_answer = None
    
#     answer_pred = extracted_answer if extracted_answer else None
#     response['answer_pred'] = answer_pred
#     return answer_pred

def extract_answer(response: dict, cache: dict) -> str:
    if 'answer_pred' in response:
        return response['answer_pred']
    
    # 捕获 \boxed{...}，考虑嵌套情况
    pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    
    try:
        matches = re.findall(pattern, str(response['content']))
        if matches:
            extracted_answer = matches[-1].strip()  # 获取最后一个匹配项并去除空格
        else:
            extracted_answer = None
            logging.debug("No matches found for pattern in content:\n" + response['content'])
    except Exception as e:
        logging.debug(f"Error {str(e)} in extracting answer in response: " + response['content'])
        extracted_answer = None

    # 如果仍然无法提取，记录日志或尝试进一步处理（可选）
    if not extracted_answer:
        logging.debug("Answer extraction failed for response content:\n" + response['content'])

    # 缓存提取结果
    answer_pred = extracted_answer if extracted_answer else None
    response['answer_pred'] = answer_pred
    return answer_pred

# def extract_again(text):
#     match = re.search(r".*[aA]nswer:\s*(\d+)", text) # 最后一个匹配 Answer: 或 answer: 后紧跟的一个正整数的字符串
#     if match:
#         return match.group(1)
#     else:
#         return extract_final(text)

# def extract_final(text):
#     pattern = r"\b\d+\b(?!.*\b\d+\b)" # 文本中最后一个独立的正整数
#     match = re.search(pattern, text, re.DOTALL)
#     return match.group(0) if match else None

def generate_single_response(example: dict, cache: dict, idx: int) -> tuple[int, int]:
    response = get_response(example, cache, idx=idx)
    answer = extract_answer(response, cache)
    if answer is None:
        logging.info(f"\nAnswer is None for problem: {example['problem'][:50]}, idx: {idx}, token used: {response['tokens']}.\n")
    return answer, response['tokens']

def generate_sampled_responses(example: dict, cache: dict, example_idx: int) -> list[tuple[str, int]]:
    responses = []
    local_cache = {example['unique_id']: cache.get(example['unique_id'], {})}

    logging.info(f"Sampling responses for problem {example_idx + 1}/500 starting with: {example['problem'][:50]}.\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS_SINGLE) as executor:
        futures = [executor.submit(generate_single_response, example, local_cache, idx) for idx in range(N_SAMPLE)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
                responses.append((answer, tokens))
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")

    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {responses}.\n\n")
    with cache_lock:
        cache.update(local_cache)
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    return responses


def is_answer_correct(answer, answer_pred):
    """
    判断答案是否正确，考虑格式差异和数值等价性。
    """
    # 1. 预处理：移除空格、标准化转义字符
    def preprocess(expr):
        # 移除多余空格
        expr = re.sub(r'\s+', '', expr)
        # 处理转义字符
        expr = expr.replace('\\', '')
        return expr

    answer = preprocess(answer)
    answer_pred = preprocess(answer_pred)

    # 2. 尝试用 SymPy 解析为数学表达式
    try:
        # 尝试将答案解析为 SymPy 表达式
        answer_sym = sp.sympify(answer, evaluate=True)
        answer_pred_sym = sp.sympify(answer_pred, evaluate=True)
        
        # 判断数学表达式是否等价
        if sp.simplify(answer_sym - answer_pred_sym) == 0:
            return True
    except (sp.SympifyError, TypeError):
        pass  # 如果解析失败，继续用字符串比较

    # 3. 字符串等价性检查（考虑可能的语义相同表达式）
    return answer == answer_pred


def calculate_bucket_accuracy(dataset: list[dict], cache: dict):

    dataset = [example for idx, example in enumerate(dataset) if idx < N_PROBLEM] # for testing

    # Gather all token counts from sampled responses
    all_token_counts = []
    logging.info(f"Sampling responses {N_SAMPLE} times for each problem in {len(dataset)} problems.\n\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(generate_sampled_responses, example, cache, idx) for idx, example in enumerate(dataset)]
        for future in concurrent.futures.as_completed(futures):
            sampled_responses = future.result()
            all_token_counts.extend([tokens for _, tokens in sampled_responses])
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
                # score = 1 if random_response[0] is not None and is_answer_correct(example['answer'], random_response[0]) else 0
                # results_by_bucket[bucket_idx].append(score)
                ## choose multiple then average
                resample_count = min(len(bucket_responses), N_SAMPLES_PER_PROBLEM)
                resampled_idx = np.random.choice(len(bucket_responses), size=resample_count, replace=False)
                resampled_responses = [bucket_responses[i] for i in resampled_idx]
                scores = [
                    1 if response[0] is not None and is_answer_correct(example['answer'], response[0]) else 0
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
    result_file = os.path.join(run_output_dir, "bucket_accuracies.json")
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
        plt.title(f"Token Count vs. Accuracy for {GEMINI_MODEL}")
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
