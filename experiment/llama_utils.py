import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import torch
import json_repair
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from experiment.utils import metrics_fn


instruction = """
You must explain why a social media message is hateful or not and then tell me your decision. You must always reply with only a JSON containing one field 'hate_speech' including a Boolean value ("True" for hate speech messages, "False" for neutral ones); and a field 'explanations' containing a list with the each message phrase and its corresponding explanation. Do not include text outside the JSON.
This is the definition of hate speech: "language characterized by offensive, derogatory, humiliating, or insulting discourse that promotes violence, discrimination, or hostility towards individuals or groups based on attributes such as race, religion, ethnicity, or gender".
The input format is: Generate step-by-step explanation for:\n<Message><input query></Message>.
The output format is:
    {{
        "hate_speech": "<Boolean>",
        "explanations": [
            {{
                "input": "<input query phrase 1>",
                "explanation": "<input query 1 phrase step-by-step explanation>"
            }},
            {{
                "input": "<input query phrase 2>",
                "explanation": "<input query 2 phrase step-by-step explanation>"
            }}
        ]
    }}
Generate step-by-step explanation for:\n<Message>{}</Message>"""

cot = [
    {
        "from": "system",
        "value": "You are an expert in explaining and detecting hate speech messages."
    }
]

seeds = [3470, 42, 3407, 366, 1008]


def transform_messages(messages):
    mapping = {"human": "user", "gpt": "assistant", "system": "system"}
    new_messages = []
    for m in messages:
        new_messages.append({
            "role": mapping.get(m["from"], m["from"]),
            "content": m["value"]
        })
    return new_messages


def get_llama_results(model_name, dataset_name, result_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", token="hf_token")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_from_disk(f"./excluded_files/dataset/{dataset_name}")
    test_set = dataset["test"]
    ground_truth_list = list(test_set["label"])

    model.eval()
    for seed in seeds:
        set_seed(seed=seed)
        output_list = []
        for p in tqdm(test_set["text"]):
            raw_message = cot + [{"from": "human", "value": instruction.format(p)}]
            if "llama" in result_name:
                formatted_message = transform_messages(raw_message)
            else:
                formatted_message = raw_message

            inputs = tokenizer.apply_chat_template(
                formatted_message,
                add_generation_prompt=True,
                return_tensors="pt",
                # padding=True,
                truncation=True,
                max_length=4096,
                return_dict=True
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    use_cache=True,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=tokenizer.pad_token_id
                )

            input_length = inputs["input_ids"].shape[1]
            result = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
            decoded_object = json_repair.loads(result)

            if isinstance(decoded_object, dict):
                val = str(decoded_object.get("hate_speech", "")).lower()
                pred = 1 if val == "true" else 0
            else:
                pred = 1
            output_list.append(pred)

        evaluate_result = metrics_fn(ground_truth_list, output_list, "macro")
        print(evaluate_result)
