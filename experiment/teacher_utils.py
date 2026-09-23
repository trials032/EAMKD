import os
import shutil
import time
import adapters
import torch
import torch.nn.functional as F
from adapters import LoRAConfig, SeqBnConfig
from datasets import load_from_disk
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from experiment.utils import metrics_fn, train_log, valid_log, save_checkpoint

lora_config = LoRAConfig(selfattn_lora=True, intermediate_lora=True, output_lora=False, r=8, alpha=2, use_gating=True)
adapter_config = SeqBnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, use_gating=True)


def get_config():
    criterion = CrossEntropyLoss()
    metrics_average = "macro"
    save_interval = 1
    train_bs = 64
    wd = 0.001
    test_bs = 256
    return criterion, metrics_average, save_interval, train_bs, wd, test_bs


class MyDataset(Dataset):
    def __init__(self, t_dataset):
        super().__init__()
        self.texts = t_dataset["text"]
        self.labels = t_dataset["label"]

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]

    def __len__(self):
        return len(self.texts)


def tch_collate_fn(tokenizer):
    def collate_fn(batch):
        texts, labels = [], []
        for text, label in batch:
            texts.append(text)
            labels.append(label)
        y = torch.LongTensor(labels).cuda()
        results = tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors='pt')
        x = {k: v.cuda() for k, v in results.items()}
        return x, y

    return collate_fn


def _validate(model, criterion, loader, metrics_average, if_correct=False):
    ground_truth_list, output_list = [], []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            logits = model(**x).logits
            loss = criterion(logits, y)

            pred = torch.argmax(F.softmax(logits, dim=-1), dim=1).squeeze()
            ground_truth_list.extend(y.cpu().tolist())
            output_list.extend(pred.cpu().tolist())
            total_loss += loss.item()
        evaluate_result = metrics_fn(ground_truth_list, output_list, metrics_average, if_correct)
    return total_loss / len(loader), evaluate_result


def train(criterion, train_set, valid_set, if_peft, checkpoint_dir, tokenizer_path, model_path, save_interval,
          metrics_average, max_epoch, batch_size, lr, wd):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    # for name, param in model.named_parameters():
    #     if "classifier" not in name:
    #         param.requires_grad = False
    if if_peft:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters without PEFT: {total_params}")
        adapters.init(model)
        model.add_adapter("lora_peft", config=lora_config)
        model.set_active_adapters("lora_peft")
        model.train_adapter("lora_peft")
        model.add_adapter("adapter_peft", config=adapter_config)
        model.set_active_adapters("adapter_peft")
        # model.train_adapter("adapter_peft")
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Trainable parameter: {name}, shape: {param.shape}")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {total_params}")
    model = model.cuda()
    train_loader = DataLoader(MyDataset(train_set), batch_size=batch_size, shuffle=True,
                              collate_fn=tch_collate_fn(tokenizer))
    valid_loader = DataLoader(MyDataset(valid_set), batch_size=batch_size, shuffle=False,
                              collate_fn=tch_collate_fn(tokenizer))
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best, best_result = False, 0
    start_time = time.perf_counter()
    for epoch in range(max_epoch):
        model.train()
        for batch_idx, (x, y) in enumerate(train_loader):
            optimizer.zero_grad()
            logits = model(**x).logits
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if batch_idx % round(len(train_loader) * 0.15) == 0:
                train_log(optimizer, loss, batch_idx, epoch, max_epoch, train_loader)

        valid_loss, valid_result = _validate(model, criterion, valid_loader, metrics_average)
        valid_log(epoch, max_epoch, valid_loss, valid_result)
        if valid_result["f1_score"] > best_result:
            best_result = valid_result["f1_score"]
            best = True

        if epoch % save_interval == 0 or best:
            save_checkpoint(model, checkpoint_dir, epoch, best, if_peft)
            best = False

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / max_epoch
    print(avg_time)


def evaluate(criterion, test_set, tokenizer_path, model_path, model_name, if_peft, metrics_average, batch_size,
             if_correct=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    if not if_peft:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    else:
        # model = AutoAdapterModel.from_pretrained(model_path).cuda()
        model = AutoModelForSequenceClassification.from_pretrained(f"./excluded_files/pretrained/{model_name}")
        adapters.init(model)
        model.add_adapter("lora_peft", config=lora_config)
        model.set_active_adapters("lora_peft")
        model.add_adapter("adapter_peft", config=adapter_config)
        model.set_active_adapters("adapter_peft")
        trainable_state_dict = torch.load(f"{model_path}.pth", weights_only=True)
        model_state_dict = model.state_dict()
        model_state_dict.update(trainable_state_dict)
        model.load_state_dict(model_state_dict)
        # model.merge_adapter("lora_peft")  # Cannot merge LoRA layer with gating
    model = model.cuda()
    test_loader = DataLoader(MyDataset(test_set), batch_size=batch_size, shuffle=False, collate_fn=tch_collate_fn(tokenizer))
    _, evaluate_result = _validate(model, criterion, test_loader, metrics_average, if_correct)
    return evaluate_result


def launch_train_evaluate(seed, model_name, dataset_name, if_train: bool, if_evaluate: bool, if_peft: bool,
                          if_ablation: str = ""):
    criterion, metrics_average, save_interval, train_bs, wd, test_bs = get_config()

    model_path = f"./excluded_files/pretrained/{model_name}"
    tokenizer_path = f"./excluded_files/pretrained/{model_name}_tokenizer"
    dataset = load_from_disk(f"./excluded_files/dataset/{dataset_name}")
    if if_peft:
        checkpoint_dir = f"./excluded_files/checkpoint/teacher_{model_name}_{dataset_name}_{seed}{if_ablation}"
        result_name = f"teacher_{model_name}_{dataset_name}{if_ablation}"
        max_epoch, lr = 10, 1e-04
    else:
        checkpoint_dir = f"./excluded_files/checkpoint/{model_name}_{dataset_name}_{seed}{if_ablation}"
        result_name = f"{model_name}_{dataset_name}{if_ablation}"
        max_epoch, lr = 10, 1e-05
    if if_train:
        train_dataset = dataset["train"].filter(lambda example: example["text"].strip())
        train(criterion=criterion, train_set=train_dataset, valid_set=dataset["valid"], if_peft=if_peft,
              checkpoint_dir=checkpoint_dir, tokenizer_path=tokenizer_path, model_path=model_path,
              save_interval=save_interval, metrics_average=metrics_average, max_epoch=max_epoch,
              batch_size=train_bs, lr=lr, wd=wd)
    if if_evaluate:
        results = evaluate(model_path=f"{checkpoint_dir}/best_model", criterion=criterion,
                           test_set=dataset["test"], tokenizer_path=tokenizer_path, model_name=model_name,
                           if_peft=if_peft, metrics_average=metrics_average, batch_size=test_bs)
        print(results)
