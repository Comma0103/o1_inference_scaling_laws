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
import copy
import sympy as sp

from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph

torch.backends.cudnn.deterministic = True

model_path_map = {
    "QwQ-32B-Preview": "/home/shaohanh/qilongma/blob/public_models/QwQ-32B-Preview",
    "Qwen2.5-32B-Instruct": "/home/shaohanh/qilongma/blob/public_models/Qwen2.5-32B-Instruct",
    "Llama-3.1-8B-ft": "/home/shaohanh/qilongma/blob/share/Llama-3.1-8B-ft-checkpoint-402",
}

# ================ config ====================
# O1_MODEL = "o1-mini"
O1_MODEL = "QwQ-32B-Preview"
CHAT_TEMPLATE_LLAMA = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'Write a response that appropriately completes the request.\\n\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
CHAT_TEMPLATE_LLAMA_H = '''
    {% if not add_generation_prompt is defined %}\n
        {% set add_generation_prompt = false %}\n
    {% endif %}\n
    {%- set ns = namespace(found=false) -%}\n
    {%- for message in messages -%}\n
        {%- if message['role'] == 'system' -%}\n
            {%- set ns.found = true -%}\n
        {%- endif -%}\n
    {%- endfor -%}\n
    {{bos_token}}{%- if not ns.found -%}\n
        {{'Write a response that appropriately completes the request.\\n\\n'}}\n
    {%- endif %}\n
    {%- for message in messages %}\n
        {%- if message['role'] == 'system' %}\n
            {{ message['content'] }}\n
        {%- else %}\n
            {%- if message['role'] == 'user' %}\n
                {{'### Instruction:\\n' + message['content'] + '\\n\\n'}}\n
            {%- else %}\n
                {{'### Response:\\n' + message['content'] + '\\n\\n'}}\n
            {%- endif %}\n
        {%- endif %}\n
    {%- endfor %}\n
    {% if add_generation_prompt %}\n
        {{'### Response:'}}\n
    {% endif %}
'''
MESSAGES_LLAMA = [
    {"role": "user", "content": None}
]
MESSAGES_QWEN = [
    {"role": "system", "content": "You are a helpful and harmless assistant. You are Qwen developed by Alibaba. You should think step-by-step."},
    {"role": "user", "content": None}
]

TEMPERATURE = 0.8
TOP_P = 0.9
MAX_MODEL_TOKENS = 32768
MAX_NEW_TOKENS = 32768 - 2048
GPU_UTIL = 0.9
N_SAMPLE = 200
N_BUCKET = 10
N_SAMPLES_PER_PROBLEM = 10

MAX_WORKERS = 1
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/MATH500/sampling/{time_str}'
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
    if 'QwQ' in O1_MODEL or 'Qwen2.5' in O1_MODEL:
        tokenizer = AutoTokenizer.from_pretrained(model_path_map[O1_MODEL], trust_remote_code=True)
        logging.info(f"Load tokenizer complete!")
        stop_token_ids = [
            tokenizer.eos_token_id, 
            tokenizer.convert_tokens_to_ids("<|end_of_text|>"),
            tokenizer.convert_tokens_to_ids("Question:")
        ]
    elif 'Llama' in O1_MODEL:
        tokenizer = llm.get_tokenizer()
        logging.info(f"Load tokenizer complete!")
        if tokenizer.chat_template is None:
            tokenizer.chat_template = CHAT_TEMPLATE_LLAMA
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            logging.info("tokenizer.chat_template is None, use pre-defined template")
        else:
            stop_token_ids = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
            logging.info("use original template")
    else:
        raise ValueError(f"Unknown model: {O1_MODEL}")
    sampling_params = SamplingParams(
        temperature=TEMPERATURE, top_p=TOP_P, max_tokens=MAX_NEW_TOKENS,
        stop_token_ids=stop_token_ids
    )
    return (llm, sampling_params), tokenizer

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

def save_cache(cache, filename):
    with open(filename, 'w') as f:
        json.dump(cache, f)


def extract_answer(response: dict, cache: dict) -> str:
    if 'answer_pred' in response:
        return response['answer_pred']

    pattern = r"\[ \\boxed\{(.*?)\} \\" # 提取最后一个 \boxed{...} 中的内容
    try:
        match = re.search(pattern, str(response['content']))
    except Exception as e:
        logging.debug(f"Error {str(e)} in extracting answer in response: " + response['content'])
    if match:
        extracted_answer = match.group(1).strip()
    else:
        logging.debug("1st answer extract failed in: \n" + response['content'])
        # extracted_answer = extract_again(response['content'])
        extracted_answer = None
    
    answer_pred = extracted_answer if extracted_answer else None
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


def batch_inference(llm, tokenizer, sampling_params, inference_batch, cache, example_id, input_encoded=False):
    start = time.time()
    if not input_encoded:
        outputs = llm.generate(inference_batch, sampling_params)
    else:
        encoded_inputs = tokenizer.batch_encode_plus(inference_batch, add_special_tokens=False)
        input_ids = encoded_inputs['input_ids']
        outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
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

def generate_sampled_responses(example: dict, model, tokenizer, cache: dict) -> list[tuple[str, int]]:
    if example['unique_id'] not in cache:
        cache[example['unique_id']] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'subject': example['subject'], 'level': example['level'], 'responses': {}}
    elif len(cache[example['unique_id']]['responses']) >= N_SAMPLE:
        logging.info(f"Cache hit for problem: {example['problem'][:50]}.")
        return [(resp['answer_pred'], resp['tokens']) for resp in cache[example['unique_id']]['responses'].values()]

    llm, sampling_params = model
    inference_batch = []

    logging.info("generating prompts for: " + example['problem'][:50] + "\n")
    if 'QwQ' in O1_MODEL or 'Qwen2.5' in O1_MODEL:
        messages = copy.deepcopy(MESSAGES_QWEN)
        messages[1]['content'] = example['problem']
    elif 'Llama' in O1_MODEL:
        messages = copy.deepcopy(MESSAGES_LLAMA)
        messages[0]['content'] = example['problem']
    else:
        raise ValueError(f"Unknown model: {O1_MODEL}")
    for idx in tqdm(range(N_SAMPLE), ncols=75):
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inference_batch.append(formatted_prompt)
    
    logging.info("evaluating: " + example['problem'][:50])
    responses = batch_inference(llm, tokenizer, sampling_params, inference_batch, cache, example['unique_id'], input_encoded=True)
    logging.info("\n")
    
    logging.info(f"Obtained answers for problem starting with: {example['problem'][:50]}.\n"
                  f"Correct answer: {example['answer']}.\n"
                  f"Obtained answers (with tokens used): {responses}.\n\n")
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
    bucket_boundaries = [round(e) for e in np.percentile(all_token_counts, np.linspace(0, 100, N_BUCKET + 1))]
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
                results_by_bucket[bucket_idx].append(average_score)
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
    model, tokenizer = load_model()
    dataset = load_math500()
    bucket_accuracies = calculate_bucket_accuracy(dataset, model, tokenizer, cache)

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

        plot_file = os.path.join(run_output_dir, f"accuracy_plot_{N_SAMPLES_PER_PROBLEM}.png")
        plt.savefig(plot_file)
        plt.close()

        logging.info(f"Accuracy plot saved to {plot_file}\n\n")
    except Exception as e:
        logging.error(f"Error generating plot: {str(e)}\n\n")

if __name__ == "__main__":
    main()