import os
import time
import adapters
import torch
from adapters import LoRAConfig, SeqBnConfig
from calflops import calculate_flops
from transformers import AutoTokenizer, AutoModelForSequenceClassification, set_seed


def IT():
    text = "a " * 128
    result = tokenizer(text, return_tensors="pt")
    x = {k: v.cuda() for k, v in result.items()}
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    num_runs = 5000
    model.eval()
    with torch.no_grad():
        model(**x)
        start_time = time.perf_counter()
        for _ in range(num_runs):
            model(**x)
        end_time = time.perf_counter()
        total_time = end_time - start_time
    peak_mem = torch.cuda.max_memory_allocated() / 1024 ** 2
    avg_inference_time = (total_time / num_runs) * 1000
    return peak_mem, avg_inference_time


def launch_cal():
    flops, _, params = calculate_flops(model=model, input_shape=(1, 128), transformer_tokenizer=tokenizer)
    peak_mem, it = IT()
    print(f"TP: {params}, MC: {peak_mem} MB, IT: {it:.2f} ms, FLOPs: {flops}")


if __name__ == "__main__":
    model_name = 'tinybert'
    model_path = f"./excluded_files/pretrained/{model_name}"
    tokenizer_path = f"./excluded_files/pretrained/{model_name}_tokenizer"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    if model_name != 'tinybert':
        adapters.init(model)
        lora_config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=False, r=8, alpha=2,
                                 use_gating=True)
        model.add_adapter("lora_peft", config=lora_config)
        model.set_active_adapters("lora_peft")
        adapter_config = SeqBnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, use_gating=True)
        model.add_adapter("adapter_peft", config=adapter_config)
        model.set_active_adapters("adapter_peft")
    model = model.cuda()
    launch_cal()
