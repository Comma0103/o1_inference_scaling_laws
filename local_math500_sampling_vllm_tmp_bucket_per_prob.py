import os
import sys
import json
import typing
import time
import logging
import threading
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
import matplotlib.pyplot as plt

from helpers.plot_helpers import plot_majority_vote_graph, plot_just_ask_nicely_graph

torch.backends.cudnn.deterministic = True

model_path_map = {
    "QwQ-32B-Preview": "/home/shaohanh/qilongma/blob/public_models/QwQ-32B-Preview",
    "Qwen2.5-32B-Instruct": "/home/shaohanh/qilongma/blob/public_models/Qwen2.5-32B-Instruct",
    "Llama-3.1-8B-ft": "/home/shaohanh/qilongma/blob/share/Llama-3.1-8B-ft-checkpoint-402",
    "Llama-3.1-8B-qwq_math_sft-random": "/home/shaohanh/qilongma/blob/share/sft_checkpoints/llama3.1_lora_4096_bsz8_reason_random",
    "Llama-3.1-8B-qwq_math_sft-long": "/home/shaohanh/qilongma/blob/share/sft_checkpoints/llama3.1_lora_4096_bsz8_reason_max",
    "Llama-3.1-8B-qwq_math_sft-short": "/home/shaohanh/qilongma/blob/share/sft_checkpoints/llama3.1_lora_4096_bsz8_reason_min",
}

# ================ config ====================
# O1_MODEL = "o1-mini"
O1_MODEL = "Llama-3.1-8B-qwq_math_sft-short"
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
N_PROBLEM = 250
N_SAMPLE = 32
N_BUCKET = 10
N_SAMPLES_PER_PROBLEM = 10

MAX_WORKERS = 1
# ================ config ====================


SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/{O1_MODEL}/MATH500/sampling/{time_str}'
# run_output_dir = '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/QwQ-32B-Preview/MATH500/sampling/12-24_04-39_copy'
run_output_dir = '/home/shaohanh/qilongma/blob/inf_scal_law/results/Llama-3.1-8B-qwq_math_sft-short/MATH500/sampling/01-02_00-40'
os.makedirs(run_output_dir, exist_ok=True)
plot_dir = os.path.join(run_output_dir, 'acc_per_prob')
os.makedirs(plot_dir, exist_ok=True)

RESPONSE_CACHE_FILENAME = f'{run_output_dir}/response_cache.json'
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile_tmp.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

cache_lock = threading.Lock()

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

def generate_sampled_responses(example: dict, model, tokenizer, cache: dict, example_idx: int) -> list[tuple[str, int]]:
    if example['unique_id'] not in cache:
        cache[example['unique_id']] = {'problem': example['problem'], 'solution': example['solution'], 'answer': example['answer'], 'subject': example['subject'], 'level': example['level'], 'responses': {}}
    elif len(cache[example['unique_id']]['responses']) >= N_SAMPLE:
        logging.info(f"Cache hit for problem: {example['problem'][:50]}.")
        return [(resp['answer_pred'], resp['tokens']) for resp in cache[example['unique_id']]['responses'].values()]

    llm, sampling_params = model
    inference_batch = []
    local_cache = {example['unique_id']: cache.get(example['unique_id'], {})}

    logging.info(f"generating prompts for problem {example_idx+1}/500 starting with: " + example['problem'][:50] + "\n")
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
    
    logging.info(f"evaluating {example_idx+1}/500 starting with: " + example['problem'][:50])
    responses = batch_inference(llm, tokenizer, sampling_params, inference_batch, local_cache, example['unique_id'], input_encoded=True)
    logging.info("\n")
    
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


def calculate_bucket_accuracy(dataset: list[dict], model, tokenizer, cache: dict):
    
    dataset = [example for idx, example in enumerate(dataset) if idx < N_PROBLEM] # for testing
    
    # Gather all token counts from sampled responses
    all_token_counts = {}
    logging.info(f"Sampling responses {N_SAMPLE} times for each problem in {len(dataset)} problems.\n\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_example = {executor.submit(generate_sampled_responses, example, model, tokenizer, cache, idx): example for idx, example in enumerate(dataset)}
        for future in concurrent.futures.as_completed(future_to_example):
            example = future_to_example[future]
            sampled_responses = future.result()
            all_token_counts[example['unique_id']] = [tokens for _, tokens in sampled_responses]
    with cache_lock:
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Calculate bucket boundaries
    logging.info(f"Calculating bucket boundaries per problem.")
    bucket_boundaries = {
        example_id: [round(e) for e in np.percentile(prob_token_counts, np.linspace(0, 100, N_BUCKET + 1))] 
        for example_id, prob_token_counts in all_token_counts.items()
    }
    logging.info(f"Bucket boundaries per problem calculated. First 10 examples: {list(bucket_boundaries.items())[:10]}.\n\n")

    # Assign responses to buckets and calculate accuracy
    results_by_bucket = {example['unique_id']: {i: 0.0 for i in range(1, N_BUCKET + 1)} for example in dataset}
    logging.info(f"Assigning responses to buckets and calculating accuracy.")
    num_missing_in_bucket = {example['unique_id']: {i: 0 for i in range(1, N_BUCKET + 1)} for example in dataset}
    example_idx = 0
    for example in tqdm(dataset, ncols=75):
        sampled_responses = generate_sampled_responses(example, model, tokenizer, cache, example_idx)
        for idx, boundary in enumerate(bucket_boundaries[example['unique_id']][:-1]):
            bucket_idx = idx + 1
            bucket_responses = [resp for resp in sampled_responses 
                                if bucket_boundaries[example['unique_id']][idx] <= resp[1] < bucket_boundaries[example['unique_id']][bucket_idx]]

            if bucket_responses:
                ## choose one response in the bucket
                # random_response = bucket_responses[np.random.randint(0, len(bucket_responses))]
                # score = 1 if random_response[0] is not None and is_answer_correct(example['answer'], random_response[0]) else 0
                # results_by_bucket[example['unique_id']][bucket_idx] = score
                ## choose multiple then average
                resample_count = min(len(bucket_responses), N_SAMPLES_PER_PROBLEM)
                resampled_idx = np.random.choice(len(bucket_responses), size=resample_count, replace=False)
                resampled_responses = [bucket_responses[i] for i in resampled_idx]
                scores = [
                    1 if response[0] is not None and is_answer_correct(example['answer'], response[0]) else 0
                    for response in resampled_responses
                ]
                average_score = np.mean(scores)
                results_by_bucket[example['unique_id']][bucket_idx] = average_score
            else:
                num_missing_in_bucket[example['unique_id']][bucket_idx] += 1
        example_idx += 1
    logging.info(f"Number of missing responses in each bucket for first 10 problems: {list(num_missing_in_bucket.items())[:10]}.\n\n")

    # Calculate and log accuracy for each bucket
    logging.info(f"Bucket accuracies per problem:\n")
    bucket_accuracies = {example['unique_id']: {i: {} for i in range(1, N_BUCKET + 1)} for example in dataset}
    for idx, (example_id, prob_results) in enumerate(results_by_bucket.items()):
        logging.info(f"Problem {idx + 1}, {example_id}: {cache[example_id]['problem'][:100 if len(cache[example_id]['problem']) > 100 else -1]}")
        for bucket, score in prob_results.items():
            bucket_accuracies[example_id][bucket] = {
                'boundary': (bucket_boundaries[example_id][bucket-1], bucket_boundaries[example_id][bucket]), 
                'accuracy': score, 
                'num_missing': num_missing_in_bucket[example_id][bucket]
            }
            logging.info(f"Bucket {bucket} ({bucket_boundaries[example_id][bucket-1]} - {bucket_boundaries[example_id][bucket]}): Accuracy {score}, Missing {num_missing_in_bucket[example_id][bucket]}")
        logging.info("\n")

    return bucket_accuracies

# Main processing
def main():
    cache = get_or_create_cache(RESPONSE_CACHE_FILENAME)
    # model, tokenizer = load_model()
    model, tokenizer = (None, None), None
    dataset = load_math500()
    bucket_accuracies = calculate_bucket_accuracy(dataset, model, tokenizer, cache)

    # Save final results
    result_file = os.path.join(run_output_dir, "bucket_accuracies_per_prob.json")
    with open(result_file, 'w') as f:
        json.dump(bucket_accuracies, f, indent=2)
    logging.info(f"\n\nFinal bucket accuracies saved to {result_file}\n\n")
    with cache_lock:
        save_cache(cache, RESPONSE_CACHE_FILENAME)

    # Plot token count vs. accuracy curve
    accuracies_all = []
    for example_id, prob_accuracies in bucket_accuracies.items():
        try:
            logging.info("Generating accuracy plot for problem: " + cache[example_id]['problem'][:100 if len(cache[example_id]['problem']) > 100 else -1])
            boundaries = [prob_accuracies[b]['boundary'] for b in prob_accuracies]
            accuracies = [prob_accuracies[b]['accuracy'] for b in prob_accuracies]
            accuracies_all.append(accuracies)
            lower_bounds = [boundary[0] for boundary in boundaries]

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(lower_bounds, accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
            plt.xscale('log', base=2)
            plt.xlabel("Token Count (Lower Boundary)")
            plt.ylabel("Accuracy")
            plt.title(f"Token Count vs. Accuracy for {O1_MODEL} on problem id: {example_id}")
            plt.grid(True)
            plt.legend()

            plot_file = os.path.join(plot_dir, f"accuracy_plot_{N_SAMPLES_PER_PROBLEM}_{example_id.replace('.json', '').replace('/', '_')}.png")
            plt.savefig(plot_file)
            plt.close()

            logging.info(f"Accuracy plot saved to {plot_file}\n")
        except Exception as e:
            logging.error(f"Error generating plot: {str(e)}\n")
    
    accuracies_all = np.array(accuracies_all)
    bucket_mean_accuracies = np.mean(accuracies_all, axis=0)

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, N_BUCKET + 1), bucket_mean_accuracies, marker='o', linestyle='-', color='b', label='Bucket Mean Accuracy over all problems')
    plt.xlabel("Bucket Index")
    plt.ylabel("Mean Accuracy")
    plt.title(f"Mean Accuracy over all problems for {O1_MODEL}")
    plt.grid(True)
    plt.legend()

    plot_file = os.path.join(plot_dir, f"bucket_mean_accuracy_plot_{N_SAMPLES_PER_PROBLEM}.png")
    plt.savefig(plot_file)
    plt.close()

    logging.info(f"Mean accuracy plot saved to {plot_file}\n")

    logging.info("All done.")

if __name__ == "__main__":
    main()
