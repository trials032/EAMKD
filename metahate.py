import os
os.environ['CUDA_VISIBLE_DEVICES'] = "5"
from experiment.llama_utils import get_llama_results


dataset_names = ["HX", "LH"]
model_name = "irlab-udc/Llama-3-8B-Distil-MetaHate"
get_llama_results(model_name, dataset_names[1], f"metahate_{dataset_names[1]}")
