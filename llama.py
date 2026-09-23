import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
from experiment.llama_utils import get_llama_results


dataset_names = ["HX", "LH"]
model_name = "unsloth/llama-3-8b-instruct-bnb-4bit"
get_llama_results(model_name, dataset_names[1], f"llama_{dataset_names[1]}")
