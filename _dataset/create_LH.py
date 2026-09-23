from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import set_seed
set_seed(seed=3470)
from create_dataset.utils import save_dataset


# Latent Hatred: A Benchmark for Understanding Implicit Hate Speech


def correct_LH(example):
    if example['label'] == 1:
        print(example['text'])
    if example['label'] == 0 or example['label'] == 1:
        example['label'] = 1
    else:
        example['label'] = 0
    return example


df = pd.read_csv("../files/original_dataset/LH/implicit_hate_v1_stg1_posts.tsv", sep='\t')
raw_set = Dataset.from_pandas(df)
raw_set = Dataset.from_dict({"text": raw_set["post"], "label": raw_set["class"]})
# print(Counter(raw_set["label"]))
raw_set = raw_set.class_encode_column("label")
raw_set = raw_set.map(correct_LH)
# split_sets = raw_set.train_test_split(test_size=0.4, stratify_by_column="label")
# raw_train_set = split_sets['train']
# split_sets = split_sets['test'].train_test_split(test_size=0.5, stratify_by_column="label")
# raw_valid_set = split_sets['train']
# raw_test_set = split_sets['test']
# dataset = DatasetDict({
#     'train': raw_train_set,
#     'valid': raw_valid_set,
#     'test': raw_test_set
# })
# folder_path = f"../files/dataset/LH"
# save_dataset(folder_path, dataset)
