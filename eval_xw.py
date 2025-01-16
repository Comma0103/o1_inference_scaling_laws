from vllm import LLM, SamplingParams
from tqdm import tqdm
import datasets
import json
import argparse
import random
import numpy as np
import re, os
import torch
from math_utils import last_boxed_only_string, remove_boxed, get_unnormalized_answer, normalize_final_answer, is_equiv, hendrycks_is_equiv

def convert_to_message(example, tokenizer):
    messages = [{"role": "user", "content": example["prompt"]}]
    example["messages"] = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return example

def load_jsonl():
    datasets_path ='qq8933/MATH500'
    dataset = datasets.load_dataset(datasets_path)
    items = []
    for item in dataset['test']:
        items.append({
            'prompt': item['problem'],
            'answer': item['answer']
        })
    print(f"load {len(items)} items")
    return items

def evaluate(eval_set):
    all_num = 0
    cor_num = 0
    output_lengths = []
    for example in eval_set:
        output_lengths.append(len(example['output'].split()))
        all_num += 1
        # extract answer from output
        match = re.search(r'\[ \\boxed\{(.*?)\} \\', example['output'])
        if match and match.group(1).strip() == example['answer']:
            cor_num += 1

    print(f"correct rate: {cor_num / all_num}")    
    print(f"average output length: {np.mean(output_lengths)}")


def verify_math_sample(model_output, ground_truth_answer):
    # model_output = model_output.split("<|assistant|>\n")[-1].strip()
    raw_answer = model_output
    # for math, more complex. We will try a few different ways to extract the answer.
    # this roughly follows 'flex em' in oe-eval-internal
    all_answers = []
    # First, try find answer in \boxed{}.
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)
    # Second, try to extract via minerva format.
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)
    # If nothing still, try to find the last latex-formatted answer
    if len(all_answers) == 0:
        dollars = [m.start() for m in re.finditer("\\$", raw_answer)]
        if len(dollars) > 1:
            # Add the answer between the second to last and last dollar sign
            answer = normalize_final_answer(raw_answer[dollars[-2] + 1 : dollars[-1]])
            all_answers.append(answer)
    # otherwise, just take the full output. Probably wont work, bit of a yolo.
    if len(all_answers) == 0:
        all_answers.append(normalize_final_answer(model_output))
    # now, compare all answers to ground truth.
    matched = False
    for answer in all_answers:
        if is_equiv(answer, ground_truth_answer):
            matched = True
            break
        elif hendrycks_is_equiv(answer, ground_truth_answer):
            matched = True
            break
    # if we got any match, we are good.
    return matched

def evaluate_for_tulu(eval_set):
    all_num = 0
    cor_num = 0
    output_lengths = []
    for example in eval_set:
        output_lengths.append(len(example['output'].split()))
        all_num += 1
        # extract answer from output
        match = verify_math_sample(example['output'], example['answer'])
        if match:
            cor_num += 1

    print(f"correct rate: {cor_num / all_num}")    
    print(f"average output length: {np.mean(output_lengths)}")

def generate_response(model_path, output_file):
    template = None

    llm = LLM(model=f"{model_path}")

    print(f"model_path: {model_path}")

    gen_kwargs_vllm = {
        "max_tokens": 8192,
        "top_p": 0.9,
        # "top_k": 50,
        "temperature": 0.8,
        "repetition_penalty": 1.0,
    }

    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is None:
        tokenizer.chat_template = template
        # tokenizer.chat_template = tokenizer.chat_template.replace("<|eot_id|>", tokenizer.eos_token)
        # tokenizer.chat_template
        gen_kwargs_vllm['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        print(f"tokenizer.chat_template: {tokenizer.chat_template}")
        print("tokenizer is None, use setted template")
    else:
        gen_kwargs_vllm['stop_token_ids'] = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|end_of_text|>")]
        print("use original template")
    
    eval_set = load_jsonl()

    eval_set = [convert_to_message(example, tokenizer) for example in eval_set]
    messages = [example['messages'] for example in eval_set]
    encoded_inputs = tokenizer.batch_encode_plus(
        messages,
        add_special_tokens=False,
    )
    input_ids = encoded_inputs['input_ids']

    sampling_params = SamplingParams(**gen_kwargs_vllm)
    torch.backends.cudnn.deterministic = True
    outputs = llm.generate(prompt_token_ids=input_ids, sampling_params=sampling_params)
    outputs_text = [x.outputs[0].text for x in outputs]
    for j in range(len(eval_set)):
        eval_set[j][f'output'] = outputs_text[j]

    with open(output_file, 'w', encoding='utf-8') as file:
        for sample in eval_set:
            json.dump(sample, file, ensure_ascii=False)
            file.write('\n')

    # evaluate the generated responses
    evaluate_for_tulu(eval_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set model name.')
    parser.add_argument('--model-path', type=str, default="/home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface", help='Dir of the model to use')
    parser.add_argument('--output-file', type=str, required=True, help='Output file')
    parser.add_argument('--only-eval', action='store_true', help='Only evaluate the generated responses')
    args = parser.parse_args()
    path_dir = args.model_path
    output_file = args.output_file
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if args.only_eval:
        print("load eval set from output file")
        with open(output_file, 'r', encoding='utf-8') as file:
            eval_set = [json.loads(line) for line in file]
        evaluate(eval_set)
    else:
        generate_response(path_dir, output_file)