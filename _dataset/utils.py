import os
import shutil
from collections import Counter
import adapters
from adapters import LoRAConfig, SeqBnConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from datasets import Dataset, load_from_disk, DatasetDict
from tqdm import tqdm

lora_config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=False, r=8, alpha=2, use_gating=True)
adapter_config = SeqBnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, use_gating=True)


def save_dataset(folder_path, dataset):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    dataset.save_to_disk(folder_path)
    print(Counter(dataset["train"]["label"]), Counter(dataset["valid"]["label"]),
          Counter(dataset["test"]["label"]), '\n')


def masked_mean_pooling(hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def masked_max_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
    masked_hidden = hidden_states.masked_fill(mask == 0, -float('inf'))
    max_pooled = torch.max(masked_hidden, dim=1).values
    return max_pooled


def distill_soft_labels(model_name, dataset_name, checkpoint, seed, batch_size: int = 256):
    dataset_dict = load_from_disk(f"../excluded_files/dataset/{dataset_name}")
    tokenizer = AutoTokenizer.from_pretrained(f"../excluded_files/pretrained/{model_name}_tokenizer")
    model = None
    for i in range(10):
        model = AutoModelForSequenceClassification.from_pretrained(f"../excluded_files/pretrained/{model_name}")
        adapters.init(model)
        model.add_adapter("lora_peft", config=lora_config)
        model.set_active_adapters("lora_peft")
        model.add_adapter("adapter_peft", config=adapter_config)
        model.set_active_adapters("adapter_peft")
        trainable_state_dict = torch.load(f"../excluded_files/checkpoint/teacher_{model_name}_{dataset_name}_{seed}/"
                                          f"checkpoint-epoch-{i}.pth", weights_only=True)
        model_state_dict = model.state_dict()
        model_state_dict.update(trainable_state_dict)
        model.load_state_dict(model_state_dict)
        if i == checkpoint:
            break
    model = model.cuda()

    distilled_dataset_dict = DatasetDict()
    model.eval()
    with torch.no_grad():
        for split_name, dataset in dataset_dict.items():
            hard_labels_list, inters_list, logits_list = [], [], []
            for i in tqdm(range(0, dataset.num_rows, batch_size)):
                texts = dataset["text"][i:i + batch_size]
                hard_labels = dataset["label"][i:i + batch_size]
                results = tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors='pt')
                x = {k: v.cuda() for k, v in results.items()}

                outputs = model(**x, output_hidden_states=True, return_dict=True)
                logits = outputs.logits
                if model_name == "bart":
                    hidden_states = outputs.decoder_hidden_states[-1]
                    # hidden_states = model.model(**x, output_hidden_states=True, return_dict=True).last_hidden_state
                else:
                    hidden_states = outputs.hidden_states[-1]
                    # hidden_states = model.roberta(**x, output_hidden_states=True, return_dict=True).last_hidden_state
                    # hidden_states = model.deberta(**x, output_hidden_states=True, return_dict=True).last_hidden_state
                inters = masked_mean_pooling(hidden_states, x["attention_mask"]).squeeze()

                hard_labels_list.extend(hard_labels)
                inters_list.extend(inters.tolist())
                logits_list.extend(logits.tolist())
            distilled_dataset = Dataset.from_dict({
                "text": dataset["text"],
                "hard_label": hard_labels_list,
                model_name + "_inter": inters_list,
                model_name + "_logit": logits_list
            })
            distilled_dataset_dict[split_name] = distilled_dataset
    return distilled_dataset_dict


def get_distilled_dataset(dataset_name, model_checkpoint, seed):
    train_dataset, valid_dataset, test_dataset = None, None, None
    for model_name, checkpoint in model_checkpoint.items():
        print(f"Distilling {model_name}...")
        t_dataset = distill_soft_labels(model_name, dataset_name, checkpoint, seed)
        if train_dataset is None:
            train_dataset = t_dataset["train"]
        else:
            train_dataset = train_dataset.add_column(model_name + "_inter", t_dataset["train"][model_name + "_inter"])
            train_dataset = train_dataset.add_column(model_name + "_logit", t_dataset["train"][model_name + "_logit"])
        if valid_dataset is None:
            valid_dataset = t_dataset["valid"]
        else:
            valid_dataset = valid_dataset.add_column(model_name + "_inter", t_dataset["valid"][model_name + "_inter"])
            valid_dataset = valid_dataset.add_column(model_name + "_logit", t_dataset["valid"][model_name + "_logit"])
        if test_dataset is None:
            test_dataset = t_dataset["test"]
        else:
            test_dataset = test_dataset.add_column(model_name + "_inter", t_dataset["test"][model_name + "_inter"])
            test_dataset = test_dataset.add_column(model_name + "_logit", t_dataset["test"][model_name + "_logit"])
    dataset_distilled = DatasetDict({
        "train": train_dataset,
        "valid": valid_dataset,
        "test": test_dataset
    })
    return dataset_distilled
