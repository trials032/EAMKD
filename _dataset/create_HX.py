import json
from collections import Counter

import pandas as pd
from datasets import Dataset, DatasetDict
from create_dataset.utils import save_dataset

# HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection:


with open('../files/original_dataset/HX/post_id_divisions.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    train_ids = data["train"]
    valid_ids = data["val"]
    test_ids = data["test"]

train_texts, train_labels = [], []
valid_texts, valid_labels = [], []
test_texts, test_labels = [], []
# tmp_text_list, tmp_label_list = [], []
with open('../files/original_dataset/HX/dataset.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
    for twitter_id, item in data.items():
        text = " ".join(item['post_tokens'])
        label = Counter([annotator['label'] for annotator in item["annotators"]]).most_common(1)[0][0]
        # tmp_text_list.append(text)
        # tmp_label_list.append(label)
        if label == 'offensive':
            print(text)
        # label = 0 if label == "normal" else 1
        # if twitter_id in train_ids:
        #     train_texts.append(text)
        #     train_labels.append(label)
        # elif twitter_id in valid_ids:
        #     valid_texts.append(text)
        #     valid_labels.append(label)
        # elif twitter_id in test_ids:
        #     test_texts.append(text)
        #     test_labels.append(label)

# print(Counter(tmp_label_list))
# raw_train_set = Dataset.from_dict({"text": train_texts, "label": train_labels})
# raw_valid_set = Dataset.from_dict({"text": valid_texts, "label": valid_labels})
# raw_test_set = Dataset.from_dict({"text": test_texts, "label": test_labels})
# dataset = DatasetDict({
#     'train': raw_train_set,
#     'valid': raw_valid_set,
#     'test': raw_test_set
# })
# folder_path = "../files/dataset/HX"
# save_dataset(folder_path, dataset)
