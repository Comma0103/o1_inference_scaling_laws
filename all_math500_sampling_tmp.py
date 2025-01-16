import time
import os, sys
import logging
import json
from datasets import load_dataset
import matplotlib.pyplot as plt
import concurrent.futures

import gemini_math500_sampling_tmp as gemini_code
import o1_math500_sampling_tmp as oai_code
import local_math500_sampling_vllm_tmp as local_code

N_PROBLEM = 250
N_BUCKET = 10
N_SAMPLES_PER_PROBLEM = 1
FIX_BUCKET_CDF = True
if FIX_BUCKET_CDF:
    BUCKET_STEP = 512

MAX_WORKERS = 10

model_results = {
    "gemini-2.0-flash-thinking-exp-1219": {
        'code': gemini_code,
        'n_problem': min(350, N_PROBLEM),
        'n_sample': 16,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gemini-2.0-flash-thinking-exp-1219/MATH500/sampling/12-30_11-20',
    },
    "gpt-4o": {
        'code': oai_code,
        'n_problem': min(500, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gpt-4o/MATH500/sampling/12-23_04-36_copy',
    },
    "gpt-4o-mini": {
        'code': oai_code,
        'n_problem': min(500, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/gpt-4o-mini/MATH500/sampling/12-23_04-37_copy',
    },
    "QwQ-32B-Preview": {
        'code': local_code,
        'n_problem': min(500, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/o1_inference_scaling_laws/results/QwQ-32B-Preview/MATH500/sampling/12-24_04-39_copy',
    },
    "Qwen2.5-32B-Instruct": {
        'code': local_code,
        'n_problem': min(425, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/blob/inf_scal_law/results/Qwen2.5-32B-Instruct/MATH500/sampling/01-07_01-14',
    },
    # "Llama-3.1-8B-ft": {
    #     'code': local_code,
    #     'n_problem': min(500, N_PROBLEM),
    #     'n_sample': 32,
    #     'n_bucket': N_BUCKET,
    #     'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
    #     'fix_bucket_cdf': FIX_BUCKET_CDF,
    #     'bucket_step': BUCKET_STEP,
    #     'run_output_dir': '/home/shaohanh/qilongma/blob/inf_scal_law/results/Llama-3.1-8B-ft/MATH500/sampling/12-24_01-59',
    # },
    "Llama-3.1-8B-qwq_math_sft-random": {
        'code': local_code,
        'n_problem': min(500, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/blob/inf_scal_law/results/Llama-3.1-8B-qwq_math_sft-random/MATH500/sampling/01-02_00-32',
    },
    "Llama-3.1-8B-qwq_math_sft-long": {
        'code': local_code,
        'n_problem': min(480, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/blob/inf_scal_law/results/Llama-3.1-8B-qwq_math_sft-long/MATH500/sampling/01-02_00-45',
    },
    "Llama-3.1-8B-qwq_math_sft-short": {
        'code': local_code,
        'n_problem': min(500, N_PROBLEM),
        'n_sample': 32,
        'n_bucket': N_BUCKET,
        'n_sample_per_problem': N_SAMPLES_PER_PROBLEM,
        'fix_bucket_cdf': FIX_BUCKET_CDF,
        'bucket_step': BUCKET_STEP,
        'run_output_dir': '/home/shaohanh/qilongma/blob/inf_scal_law/results/Llama-3.1-8B-qwq_math_sft-short/MATH500/sampling/01-02_00-40',
    },
}

SAVE_DIR = f'results'
timestamp = time.time()
time_str = time.strftime('%m-%d_%H-%M-%S', time.localtime(timestamp))
run_output_dir = f'{SAVE_DIR}/all_models/MATH500/sampling/{time_str}'
os.makedirs(run_output_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{run_output_dir}/logfile_tmp.log'),
        logging.StreamHandler(sys.stdout)
    ]
)


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

def get_model_result(model_name, model_info, dataset):
    code = model_info['code']
    if code == gemini_code:
        code.GEMINI_MODEL = model_name
    elif code == oai_code:
        code.O1_MODEL = model_name
    elif code == local_code:
        code.O1_MODEL = model_name
    code.N_PROBLEM = model_info['n_problem']
    code.N_SAMPLE = model_info['n_sample']
    code.N_BUCKET = model_info['n_bucket']
    code.N_SAMPLE_PER_PROBLEM = model_info['n_sample_per_problem']
    code.FIX_BUCKET_CDF = model_info['fix_bucket_cdf']
    if code.FIX_BUCKET_CDF:
        code.BUCKET_STEP = model_info['bucket_step']
    code.run_output_dir = model_info['run_output_dir']
    code.RESPONSE_CACHE_FILENAME = f'{code.run_output_dir}/response_cache.json'
    # code.logging.basicConfig(level=logging.WARNING)

    cache = code.get_or_create_cache(code.RESPONSE_CACHE_FILENAME)

    if code == local_code:
        # model, tokenizer = code.load_model()
        model, tokenizer = (None, None), None
        bucket_accuracies = code.calculate_bucket_accuracy(dataset, model, tokenizer, cache)
    else:
        bucket_accuracies = code.calculate_bucket_accuracy(dataset, cache)

    # Save final results
    result_file = os.path.join(code.run_output_dir, f"bucket_accuracies{'_fix_bucket_cdf' if code.FIX_BUCKET_CDF else ''}_{code.N_SAMPLES_PER_PROBLEM}.json")
    with open(result_file, 'w') as f:
        json.dump(bucket_accuracies, f, indent=2)
    logging.info(f"\n\nFinal bucket accuracies saved to {result_file}\n\n")
    with code.cache_lock:
        code.save_cache(cache, code.RESPONSE_CACHE_FILENAME)
    
    logging.info(f"Model {model_name} result load done, generating accuracy plot...\n\n")
    boundaries = [bucket_accuracies[b]['boundary'] for b in bucket_accuracies]
    accuracies = [bucket_accuracies[b]['accuracy'] for b in bucket_accuracies]
    lower_bounds = [boundary[0] for boundary in boundaries]
    upper_bounds = [boundary[1] for boundary in boundaries]

    return model_name, lower_bounds, upper_bounds, accuracies

def main():
    dataset = load_math500()

    plt.figure(figsize=(18, 8))

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(model_results), MAX_WORKERS)) as executor:
        futures = [executor.submit(get_model_result, model_name, model_info, dataset) for model_name, model_info in model_results.items()]
        for future in concurrent.futures.as_completed(futures):
            try:
                model_name, lower_bounds, upper_bounds, accuracies = future.result()
                plt.plot(lower_bounds, accuracies, marker='o', linestyle='-', label=f'Accuracy for {model_name}')
            except Exception as e:
                logging.exception(f"Error processing result: {e}.")
    
    # plt.xscale('log', base=2)
    plt.xticks(lower_bounds, [f'<{upper_bound}' for upper_bound in upper_bounds], rotation=45)
    plt.xlabel("Token Count (Lower Boundary)")
    plt.ylabel("Accuracy")
    plt.title(f"Token Count vs. Accuracy for {len(model_results)} models")
    plt.grid(True)
    plt.legend()

    plot_file = os.path.join(run_output_dir, f"accuracy_plot_all{'_fix_bucket_cdf' if FIX_BUCKET_CDF else ''}_{N_SAMPLES_PER_PROBLEM}.png")
    plt.savefig(plot_file)
    plt.close()

    logging.info(f"Accuracy plot saved to {plot_file}\n\n")


if __name__ == '__main__':
    main()