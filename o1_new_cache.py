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

Important: You should try your best to use around {token_limit} tokens in your reasoning steps.
If you feel like you are finished early, spend the extra tokens trying to double check your work until you are absolutely sure that you have the correct answer.

Here's the problem:

{problem}

Think step by step to solve this problem, use around {token_limit} tokens in your reasoning, and provide the final answer with "the answer is (X)" where X is a single integer.
"""
EXTRACT_MODEL = "gpt-4o-mini"
OPENAI_EXTRACT_CLIENT = Openai(apis=API_INFOS[model_name_map[EXTRACT_MODEL]])
TEMPERATURE = 0.8
TOP_P = 0.9

SHADE_REGIONS = True
RUN_FULL_RANGE = False
MAX_WORKERS = 32
MAX_WORKERS_SINGLE = 10
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/AIME/{time_str}'
os.makedirs(run_output_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

cache_lock = threading.Lock()


def load_2024_dataset() -> list[dict]:
    """
    Load the dataset of problems.

    Returns:
        list[dict]: The dataset of problems.
    """
    dataset_original = load_dataset("AI-MO/aimo-validation-aime")

    # Filter out problems that are not from 2024
    dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])

    logging.debug(f"Filtered dataset size: {len(dataset)}.")
    assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
    return dataset


def get_or_create_cache(filename: str) -> dict[str, typing.Any]:
    """
    Get the cache if it exists, otherwise create it.

    Args:
        filename (str): The filename of the cache to get or create.

    Returns:
        dict: The cache.
    """
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


def get_response(example: dict, token_limit: int, cache: dict, idx: int = 0, N: int = 0) -> dict:
    """
    Get a response from the model.

    Args:
        example (dict): The problem to process. {'id': int64, 'problem': str, 'solution': str, 'answer': str, 'url': str}
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        idx (int, optional): The index of the response to process. Defaults to 0.

    Returns:
        dict: The response from the model.
    """

    # if idx > 0:
    #     cache_key = f"{O1_MODEL}_{PROMPT}_{problem}_{token_limit}_{idx}"
    # else:
    #     cache_key = f"{O1_MODEL}_{PROMPT}_{problem}_{token_limit}"
    # cache_key = f"{token_limit}_{problem}_{idx}"
    # if cache_key in cache:
    #     logging.debug(f"Cache hit for problem: {problem[:50]}. idx: {idx}. Requested tokens: {token_limit}.")
    #     return cache[cache_key]
    with cache_lock:
        if str(example['id']) not in cache or not cache[str(example['id'])]:
            cache[str(example['id'])] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'responses': {}}
        elif str(token_limit) in cache[str(example['id'])]['responses'] and str(idx) in cache[str(example['id'])]['responses'][str(token_limit)]:
            logging.debug(f"Cache hit for problem: {example['problem'][:50]}. idx: {idx}. Requested tokens: {token_limit}.")
            return cache[str(example['id'])]['responses'][str(token_limit)][str(idx)]
    
    formatted_prompt = PROMPT.format(problem=example['problem'], token_limit=token_limit)
    logging.debug(f"Requesting {token_limit} tokens for problem starting with: {example['problem'][:50]} running {idx} of {N} times.")
    # response = OPENAI_CLIENT.default_client.chat.completions.create(
    #     model=O1_MODEL,
    #     messages=[{"role": "user", "content": formatted_prompt}]
    # )
    response = OPENAI_CLIENT.call(content=formatted_prompt, return_completion=True)
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    # cache[cache_key] = result
    if str(token_limit) not in cache[str(example['id'])]['responses']:
        cache[str(example['id'])]['responses'][str(token_limit)] = {}
    cache[str(example['id'])]['responses'][str(token_limit)][str(idx)] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {example['problem'][:50]}. Requested tokens: {token_limit}.")
    return result


# def extract_answer(response_content: str, cache: dict) -> int:
#     """
#     Extract the final integer answer from the response content.

#     Args:
#         response_content (str): The response content to extract the answer from.
#         cache (dict): The cache to use for storing responses.

#     Returns:
#         int: The final integer answer.
#     """
#     cache_key = f"extract_answer_{response_content}"
#     if cache_key in cache:
#         return cache[cache_key]

#     extraction_prompt = f"""
#     Extract the final integer answer from the following problem solution. 
#     Return only the integer, nothing else.

#     Solution:
#     {response_content}

#     Final answer (integer only):
#     """
    
#     # extraction_response = OPENAI_EXTRACT_CLIENT.default_client.chat.completions.create(
#     #     model=EXTRACT_MODEL,
#     #     messages=[{"role": "user", "content": extraction_prompt}]
#     # )
#     extraction_response = OPENAI_EXTRACT_CLIENT.call(content=extraction_prompt, max_tokens=128, return_completion=True)
#     extracted_answer = extraction_response.choices[0].message.content.strip()
#     try:
#         result = int(extracted_answer)
#     except ValueError:
#         result = None
    
#     cache[cache_key] = result
#     return result

def extract_answer(response: dict, cache: dict) -> int:
    """
    Extract the final integer answer from the response content.

    Args:
        response (dict): The response to extract the answer from. {'content': str, 'tokens': int}
        cache (dict): The cache to use for storing responses.

    Returns:
        int: The final integer answer.
    """
    # cache_key = f"extract_answer_{response_content}"
    # if cache_key in cache:
    #     return cache[cache_key]
    if 'answer_pred' in response:
        return int(response['answer_pred'])

    pattern = r"[aA]nswer is \(?(\d+)\)?" # 在文本中找到以 answer is 开头，后面紧跟的一个正整数的字符串
    try:
        matches = re.findall(pattern, str(response['content']))
        if matches:
            extracted_answer = matches[-1].strip()  # 获取最后一个匹配项并去除空格
        else:
            logging.debug("No matches found in 1st answer extract for pattern in content:\n" + response['content'])
            extracted_answer = extract_again(response['content'])
    except Exception as e:
        logging.debug(f"Error {str(e)} in extracting answer in response: " + response['content'])
        extracted_answer = None

    # 如果仍然无法提取，记录日志或尝试进一步处理（可选）
    if not extracted_answer:
        logging.debug("Answer extraction failed for response content:\n" + response['content'])

    # 缓存提取结果
    answer_pred = extracted_answer if int(extracted_answer) else None
    # cache[cache_key] = answer_pred
    response['answer_pred'] = answer_pred

    return answer_pred

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*(\d+)', text) # 最后一个匹配 Answer: 或 answer: 后紧跟的一个正整数的字符串
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b\d+\b(?!.*\b\d+\b)" # 文本中最后一个独立的正整数
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def generate_single_response(example: dict, token_limit: int, cache: dict, idx: int, N: int) -> tuple[int, int]:
    """
    Get a single response for a problem.

    Args:
        example (dict): The problem to process. {'id': int64, 'problem': str, 'solution': str, 'answer': str, 'url': str}
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        idx (int): The index of the response to process.

    Returns:
        tuple[int, int]: A tuple containing the answer and the number of tokens used.
    """
    response = get_response(example, token_limit, cache, idx=idx, N=N)
    answer = extract_answer(response, cache)
    if answer is None:
        logging.info(f"Answer is None for problem: {example['problem']}, token limit: {token_limit}, idx: {idx}.")
        answer = 0
    return answer, response['tokens']


def process_single_example(example: dict, token_limit: int, cache: dict, N: int, max_workers: int = 10) -> tuple[float, int]:
    """
    Process a single example by running the model N times and then taking the majority vote.

    Args:
        example (dict): The problem to process.
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        N (int): The number of times to run the model.

    Returns:
        tuple[bool, int]: A tuple containing the majority vote result and the total number 
        of tokens used.
    """
    answers = []
    tokens_list = []
    total_tokens = 0
    local_cache = {str(example['id']): cache.get(str(example['id']), {})}

    logging.info(f"Sampling responses for problem {example['id'] + 1}/30 starting with: {example['problem'][:50]}.\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_response, example, token_limit, local_cache, idx, N) for idx in range(N)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
                answers.append(answer)
                tokens_list.append(tokens)
                total_tokens += tokens
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")
                # answer, tokens = 0, 0
    
    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {[(answer, tokens) for answer, tokens in zip(answers, tokens_list)]}.\n")
    with cache_lock:
        cache.update(local_cache)
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Compute majority vote
    majority_answers = statistics.multimode(answers)

    score = 0
    if int(example['answer']) in majority_answers:
        # If the majority answer is in the correct answer, we consider it correct.
        # If there are multiple majority answers, we give partial credit to preserve
        # determinism.
        score = 1 / len(majority_answers)
        majority_answer = None

    return score, total_tokens


def run_experiments(dataset: list[dict], cache: dict[str, typing.Any], token_limit: int, N: int, max_workers: int = 32, max_workers_single: int = 10) -> tuple[float, float]:
    """
    Run experiments given the token limit and return results.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
        token_limit (int): The token limit for the model.
        N (int): The number of times to run the model.

    Returns:
        tuple[float, float]: A tuple containing the accuracy and average tokens used.
    """
    total_score = 0
    actual_tokens_used = []
    logging.info(f"Requested token limit: {token_limit}. Run model {N} times.\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(process_single_example, example, token_limit, cache, N, max_workers_single) for example in dataset]

        for future in concurrent.futures.as_completed(futures):
            score, tokens = future.result()
            if score > 0:
                total_score += score
            actual_tokens_used.append(tokens)
        
        with cache_lock:
            save_cache(cache, RESPONSE_CACHE_FILENAME)
    
    accuracy = total_score / len(dataset)
    avg_tokens_used = np.mean(actual_tokens_used)
    logging.info(f"Requested token limit: {token_limit}. Run model {N} times. Accuracy: {accuracy}. Average tokens used: {avg_tokens_used}.\n\n")
    return accuracy, avg_tokens_used


def run_majority_vote_inference_experiments(dataset: list[dict], cache: dict[str, typing.Any], shade_regions: bool = False, max_workers: int = 32, max_workers_single: int = 10) -> None:
    """
    Run experiments and create graphs that include majority vote extending over 2^14 tokens 
    for reasoning. We observe that models stop using more tokens even when asked to around 2^11.
    We solve this by doing repeated sampling and then taking the mode of the answers for all 
    queries above 2^11. This is not perfect, but still seems to help a bit.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
        shade_regions (bool, optional): determines whether we include the plot with shaded 
        regions describing the different strategies. If False, it generates the headline 
        reconstruction plot of the o1 inference-time scaling laws.
    """
    logging.info(f"Start running majority vote experiments.")
    logging.info(f"Shade regions: {shade_regions}.")

    if shade_regions:
        token_limits = [2**i for i in range(4, 19)]
    else:
        token_limits = [2**i for i in range(4, 15)]
    logging.info(f"Token limits: {token_limits}.\n\n")

    results = []
    for token_limit in tqdm(token_limits, ncols=75):
        logging.info(f"Running experiments for token limit: {token_limit}.")
        actual_token_limit = min(2**11, token_limit)
        # We run the experiment N times for each token limit
        N = token_limit // actual_token_limit
        accuracy, avg_tokens_used = run_experiments(dataset, cache, actual_token_limit, N, max_workers, max_workers_single)
        result = {
            'token_limit': token_limit,
            'actual_token_limit': actual_token_limit,
            'N': N,
            'accuracy': accuracy,
            'avg_tokens_used': avg_tokens_used
        }
        results.append(result)

    plot_majority_vote_graph(results, shade_regions, run_output_dir, run_output_dir)


def run_just_ask_nicely_experiments(dataset: list[dict], cache: dict[str, typing.Any], run_full_range: bool = False, max_workers: int = 32, max_workers_single: int = 10) -> None:
    """
    Run experiments where we ask the model to use more tokens by asking it to use more tokens nicely.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        cache (dict[str, typing.Any]): The cache to use for storing responses.
    """
    logging.info(f"Start running just ask nicely experiments.")
    logging.info(f"Run full range: {run_full_range}.")

    token_limits = [2**i for i in range(4, 12)]
    if run_full_range:
        token_limits = [2**i for i in range(20)]
    logging.info(f"Token limits: {token_limits}.\n\n")

    results = []
    for token_limit in tqdm(token_limits, ncols=75):
        logging.info(f"Running experiments for token limit: {token_limit}.")
        accuracy, avg_tokens_used = run_experiments(dataset, cache, token_limit, 1, max_workers, max_workers_single)
        result = {
            'token_limit': token_limit,
            'accuracy': accuracy,
            'avg_tokens_used': avg_tokens_used
        }
        results.append(result)
    
    plot_just_ask_nicely_graph(results, run_full_range, run_output_dir, run_output_dir)


dataset = load_2024_dataset()
# print(len(dataset), dataset[0])
cache = get_or_create_cache(RESPONSE_CACHE_FILENAME)
run_majority_vote_inference_experiments(dataset, cache, SHADE_REGIONS, MAX_WORKERS, MAX_WORKERS_SINGLE)
run_just_ask_nicely_experiments(dataset, cache, RUN_FULL_RANGE, MAX_WORKERS, MAX_WORKERS_SINGLE)
