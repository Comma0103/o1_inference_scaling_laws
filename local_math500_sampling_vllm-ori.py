from vllm import LLM, SamplingParams
from tqdm import tqdm
import datasets
import json
import argparse
import random
import numpy as np
import re
import torch

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
    for example in eval_set:
        all_num += 1
        # extract answer from output
        match = re.search(r'\[ \\boxed\{(.*?)\} \\', example['output'])
        if match and match.group(1).strip() == example['answer']:
            cor_num += 1
    print(f"correct rate: {cor_num / all_num}")

def generate_response(model_path, output_file):
    template = "{% if not add_generation_prompt is defined %}\n{% set add_generation_prompt = false %}\n{% endif %}\n{%- set ns = namespace(found=false) -%}\n{%- for message in messages -%}\n    {%- if message['role'] == 'system' -%}\n        {%- set ns.found = true -%}\n    {%- endif -%}\n{%- endfor -%}\n{{bos_token}}{%- if not ns.found -%}\n{{'Write a response that appropriately completes the request.\\n\\n'}}\n{%- endif %}\n{%- for message in messages %}\n    {%- if message['role'] == 'system' %}\n{{ message['content'] }}\n    {%- else %}\n        {%- if message['role'] == 'user' %}\n{{'### Instruction:\\n' + message['content'] + '\\n\\n'}}\n        {%- else %}\n{{'### Response:\\n' + message['content'] + '\\n\\n'}}\n        {%- endif %}\n    {%- endif %}\n{%- endfor %}\n{% if add_generation_prompt %}\n{{'### Response:'}}\n{% endif %}"
    template = '''
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
    # messages = tokenizer.apply_chat_template(messages, tokenize=False)

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
    evaluate(eval_set)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Set model name.')
    parser.add_argument('--model-path', type=str, default="/home/lidong1/jianglingjie/LLama-Factory/model_checkpoint/huggingface", help='Dir of the model to use')
    parser.add_argument('--output-file', type=str, required=True, help='Output file')
    args = parser.parse_args()
    path_dir = args.model_path
    output_file = args.output_file

    generate_response(path_dir, output_file)