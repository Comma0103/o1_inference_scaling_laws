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

choices = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]


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

shade_regions = True
run_full_range = False
max_workers = 32
max_workers_single = 10

DATA_PATH = "/home/shaohanh/qilongma/blob/datasets/MMLU-Pro/data"
NUM_PROBLEM_PER_CATEGORY = 100
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/MMLU_Pro/{time_str}'
os.makedirs(run_output_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


# def load_2024_dataset() -> list[dict]:
#     """
#     Load the dataset of problems.

#     Returns:
#         list[dict]: The dataset of problems.
#     """
#     dataset_original = load_dataset("AI-MO/aimo-validation-aime")

#     # Filter out problems that are not from 2024
#     dataset = dataset_original["train"].filter(lambda example: "2024" in example["url"])

#     logging.debug(f"Filtered dataset size: {len(dataset)}.")
#     assert len(dataset) == 30, f"Expected 30 problems after filtering by 2024, but found {len(dataset)}"
#     return dataset

def load_mmlu_pro() -> list[dict]:
    """
    Load the dataset of problems.

    Returns:
        list[dict]: The dataset of problems.
    """
    if DATA_PATH == "hf_hub":
        dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    else:
        dataset = load_dataset("parquet", data_dir=DATA_PATH)
    test_df, val_df = dataset["test"], dataset["validation"]
    test_df = preprocess(test_df)
    val_df = preprocess(val_df)
    return test_df, val_df

def preprocess(test_df) -> list[dict]:
    res_df = []
    for each in test_df:
        options = []
        for opt in each["options"]:
            if opt == "N/A":
                continue
            options.append(opt)
        each["options"] = options
        res_df.append(each)
    return res_df

def filter_mmlu_pro(dataset: list[dict], num_problem_per_category: int = 10) -> tuple[list[dict], list[str]]:
    """
    Filter the dataset to only include the first num_problem_per_category problems from each category.

    Args:
        dataset (list[dict]): The dataset of problems.
        num_problem_per_category (int, optional): The number of problems to include per category. Defaults to 10.

    Returns:
        list[dict]: The filtered dataset.
        list[str]: The list of categories.
    """
    filtered_dataset = []
    category_counts = {}
    for example in dataset:
        category = example["category"]
        if category not in category_counts:
            category_counts[category] = 0
        if category_counts[category] < num_problem_per_category:
            filtered_dataset.append(example)
            category_counts[category] += 1
    return filtered_dataset, list(category_counts.keys())

def select_by_category(df, subject):
    res = []
    for each in df:
        if each["category"] == subject:
            res.append(each)
    return res


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

def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)


def get_response(problem: str, token_limit: int, cache: dict, idx: int = 0, N: int = 0) -> dict:
    """
    Get a response from the model.

    Args:
        problem (str): The problem to process.
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
    cache_key = f"{token_limit}_{problem}_{idx}"
    if cache_key in cache:
        logging.debug(f"Cache hit for problem: {problem[:50]}. idx: {idx}. Requested tokens: {token_limit}.")
        return cache[cache_key]
    
    formatted_prompt = PROMPT.format(problem=problem, token_limit=token_limit)
    logging.debug(f"Requesting {token_limit} tokens for problem starting with: {problem[:50]} running {idx} of {N} times.")
    # response = OPENAI_CLIENT.default_client.chat.completions.create(
    #     model=O1_MODEL,
    #     messages=[{"role": "user", "content": formatted_prompt}]
    # )
    response = OPENAI_CLIENT.call(content=formatted_prompt, max_tokens=token_limit, return_completion=True)
    result = {
        'content': response.choices[0].message.content,
        'tokens': response.usage.completion_tokens
    }
    cache[cache_key] = result
    logging.debug(f"Received {result['tokens']} tokens for problem starting with: {problem[:50]}. Requested tokens: {token_limit}.")
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

def extract_answer(response_content: str, cache: dict) -> int:
    """
    Extract the final integer answer from the response content.

    Args:
        response_content (str): The response content to extract the answer from.
        cache (dict): The cache to use for storing responses.

    Returns:
        int: The final answer key letter.
    """
    cache_key = f"extract_answer_{response_content}"
    if cache_key in cache:
        return cache[cache_key]

    pattern = r"answer is \(?([A-J])\)?" # 在文本中找到以 answer is 开头，后面紧跟一个范围为 A-J 的字母（可选地被括号 () 包裹）的匹配项，并捕获字母作为结果。
    try:
        match = re.search(pattern, str(response_content))
    except Exception as e:
        logging.debug(f"Error {str(e)} in extracting answer in response: " + str(response_content))
    if match:
        extracted_answer = match.group(1)
    else:
        logging.info("1st answer extract failed in: \n" + str(response_content))
        extracted_answer = extract_again(response_content)
    if extracted_answer is not None:
        result = int(extracted_answer)
    else:
        result = None
    
    cache[cache_key] = result
    return result

def extract_again(text):
    match = re.search(r'.*[aA]nswer:\s*([A-J])', text) # 最后一个匹配 Answer: 或 answer: 后紧跟一个范围为 A-J 的字母，并捕获该字母作为结果。
    if match:
        return match.group(1)
    else:
        return extract_final(text)

def extract_final(text):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)" # 文本中最后一个独立的大写字母（范围 A-J）
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(0)
    else:
        return None


def generate_single_response(example: dict, token_limit: int, cache: dict, idx: int, N: int) -> tuple[int, int]:
    """
    Get a single response for a problem.

    Args:
        example (dict): The problem to process.
        token_limit (int): The token limit for the model.
        cache (dict): The cache to use for storing responses.
        idx (int): The index of the response to process.

    Returns:
        tuple[int, int]: A tuple containing the answer and the number of tokens used.
    """
    response = get_response(example['problem'], token_limit, cache, idx=idx, N=N)
    answer = extract_answer(response['content'], cache)
    if answer is None:
        logging.info(f"Answer is None for problem: {example['problem']}")
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
    total_tokens = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_response, example, token_limit, cache, idx, N) for idx in range(N)]

        for future in concurrent.futures.as_completed(futures):
            try:
                answer, tokens = future.result()
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")
                answer, tokens = 0, 0

            answers.append(answer)
            total_tokens += tokens
    
    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers: {sorted(answers)}.\n")

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
    subject = dataset[0]['category']

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:

        futures = [executor.submit(process_single_example, example, token_limit, cache, N, max_workers_single) for example in dataset]

        for future in concurrent.futures.as_completed(futures):
            score, tokens = future.result()
            if score > 0:
                total_score += score
            actual_tokens_used.append(tokens)
        
        save_cache(cache, RESPONSE_CACHE_FILENAME.replace(".json", f"_{subject}.json"))
    
    accuracy = total_score / len(dataset)
    avg_tokens_used = np.mean(actual_tokens_used)
    logging.info(f"Requested token limit: {token_limit}. Run model {N} times. Accuracy: {accuracy}. Average tokens used: {avg_tokens_used}.\n\n")
    return accuracy, avg_tokens_used


def run_majority_vote_inference_experiments(dataset: list[dict], subjects: list[str], shade_regions: bool = False, max_workers: int = 32, max_workers_single: int = 10) -> None:
    """
    Run experiments and create graphs that include majority vote extending over 2^14 tokens 
    for reasoning. We observe that models stop using more tokens even when asked to around 2^11.
    We solve this by doing repeated sampling and then taking the mode of the answers for all 
    queries above 2^11. This is not perfect, but still seems to help a bit.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        subjects (list[str]): The list of subjects to run the experiments on.
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
    logging.info(f"Token limits: {token_limits}.\n\n\n")

    results = []
    for token_limit in tqdm(token_limits, ncols=75):
        logging.info(f"Running experiments for token limit: {token_limit}.")
        actual_token_limit = min(2**11, token_limit)
        # We run the experiment N times for each token limit
        N = token_limit // actual_token_limit
        logging.info(f"Requested token limit: {actual_token_limit}. Run model {N} times.\n\n")

        for subject in sorted(subjects):
            logging.info(f"Running experiments for subject: {subject}.\n")
            dataset_subject = select_by_category(dataset, subject)
            cache = get_or_create_cache(RESPONSE_CACHE_FILENAME.replace(".json", f"_{subject}.json"))
            accuracy, avg_tokens_used = run_experiments(dataset_subject, cache, actual_token_limit, N, max_workers, max_workers_single)
            result = {
                'subject': subject,
                'token_limit': token_limit,
                'actual_token_limit': actual_token_limit,
                'N': N,
                'accuracy': accuracy,
                'avg_tokens_used': avg_tokens_used
            }
            results.append(result)

    for subject in sorted(subjects):
        results_subject = [r for r in results if r['subject'] == subject]
        plot_majority_vote_graph(results_subject, shade_regions, run_output_dir, run_output_dir, subject)


def run_just_ask_nicely_experiments(dataset: list[dict], subjects: list[str], run_full_range: bool = False, max_workers: int = 32, max_workers_single: int = 10) -> None:
    """
    Run experiments where we ask the model to use more tokens by asking it to use more tokens nicely.

    Args:
        dataset (list[dict]): The dataset of problems to run the experiments on.
        subjects (list[str]): The list of subjects to run the experiments on.
        run_full_range (bool, optional): Whether to run the full range of token limits. Defaults to False.
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


test_dataset, _ = load_mmlu_pro()
# print(len(test_dataset), test_dataset[0])
test_dataset, subjects = filter_mmlu_pro(test_dataset, NUM_PROBLEM_PER_CATEGORY)

run_majority_vote_inference_experiments(test_dataset, subjects, shade_regions, max_workers, max_workers_single)
run_just_ask_nicely_experiments(test_dataset, subjects, run_full_range, max_workers, max_workers_single)
