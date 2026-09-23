import os
import shutil
import time
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from experiment.utils import train_log, valid_log, metrics_fn, save_checkpoint, masked_mean_pooling


def get_config():
    metrics_average = "macro"
    save_interval = 1
    max_epoch = 10
    train_bs = 64
    lr = 1e-05
    wd = 0.001
    test_bs = 512
    return metrics_average, save_interval, max_epoch, train_bs, lr, wd, test_bs


class MyDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.text = dataset["text"]
        self.hard_label = dataset["hard_label"]
        self.roberta_inter = torch.FloatTensor(dataset["roberta_inter"])
        self.deberta_inter = torch.FloatTensor(dataset["deberta_inter"])
        self.bart_inter = torch.FloatTensor(dataset["bart_inter"])
        self.roberta_logit = torch.FloatTensor(dataset["roberta_logit"])
        self.deberta_logit = torch.FloatTensor(dataset["deberta_logit"])
        self.bart_logit = torch.FloatTensor(dataset["bart_logit"])

    def __getitem__(self, index):
        # text = self.text[index]
        # hard_label = self.hard_label[index]
        # tch_inter = [self.roberta_inter[index], self.deberta_inter[index], self.bart_inter[index]]
        # tch_logit = [self.roberta_logit[index], self.deberta_logit[index], self.bart_logit[index]]
        # return text, hard_label, tch_inter, tch_logit
        return (
            self.text[index],
            self.hard_label[index],
            self.roberta_inter[index],
            self.deberta_inter[index],
            self.bart_inter[index],
            self.roberta_logit[index],
            self.deberta_logit[index],
            self.bart_logit[index]
        )

    def __len__(self):
        return len(self.text)


def std_collate_fn(tokenizer):
    def collate_fn(batch):
        # texts, hard_labels, tch1_inters, tch2_inters, tch3_inters, tch_logits = [], [], [], [], [], []
        texts, labels, r_int, d_int, b_int, r_log, d_log, b_log = zip(*batch)
        # for text, hard_label, tch_inter, tch_logit in batch:
        #     texts.append(text)
        #     hard_labels.append(hard_label)
        #     tch1_inters.append(tch_inter[0])
        #     tch2_inters.append(tch_inter[1])
        #     tch3_inters.append(tch_inter[2])
        #     tch_logits.append(tch_logit)
        # y = torch.LongTensor(hard_labels).cuda()
        # h = [torch.Tensor(tch1_inters).cuda(), torch.Tensor(tch2_inters).cuda(), torch.Tensor(tch3_inters).cuda()]
        # z = torch.Tensor(tch_logits).cuda()
        # results = tokenizer(texts, padding=True, truncation=True, max_length=192, return_tensors='pt')
        # x = {k: v.cuda() for k, v in results.items()}
        # return x, y, h, z
        results = tokenizer(list(texts), padding=True, truncation=True, max_length=192, return_tensors='pt')
        x = {k: v.cuda() for k, v in results.items()}
        y = torch.LongTensor(labels).cuda()
        h = [torch.stack(r_int), torch.stack(d_int), torch.stack(b_int)]
        z = torch.stack([torch.stack(r_log), torch.stack(d_log), torch.stack(b_log)], dim=1)
        return x, y, h, z

    return collate_fn


def _validate(model, criterion, criterion_type, loader, metrics_average, if_correct=False):
    ground_truth_list, output_list = [], []
    total_loss = 0
    model.eval()
    with torch.no_grad():
        for batch_idx, (x, y, h, z) in enumerate(loader):
            # x = {k: v.cuda() for k, v in x.items()}
            # y = y.cuda()
            h = [item.cuda() for item in h]
            z = z.cuda()

            outputs = model(**x, output_hidden_states=True, return_dict=True)
            if criterion_type in ("EAMKDLoss"):
                inters = masked_mean_pooling(outputs.hidden_states[-1], x["attention_mask"])
                loss = criterion(outputs.logits, inters, y, h, z)
            else:
                loss = criterion(outputs.logits, y, z)

            pred = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=1).squeeze()
            ground_truth_list.extend(y.cpu().tolist())
            output_list.extend(pred.cpu().tolist())
            total_loss += loss.item()
        evaluate_result = metrics_fn(ground_truth_list, output_list, metrics_average, if_correct)
    return total_loss / len(loader), evaluate_result


def train(criterion, train_set, valid_set, checkpoint_dir, tokenizer_path, model_path, criterion_type, save_interval,
          metrics_average, max_epoch, batch_size, lr, wd):
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    train_loader = DataLoader(MyDataset(train_set), batch_size=batch_size, shuffle=True,
                              collate_fn=std_collate_fn(tokenizer))
    valid_loader = DataLoader(MyDataset(valid_set), batch_size=batch_size, shuffle=False,
                              collate_fn=std_collate_fn(tokenizer))
    if criterion_type in ("EAMKDLoss"):
        optimizer = AdamW([{'params': model.parameters(), "lr": lr},
                           {'params': criterion.parameters(), "lr": lr * 10}], weight_decay=wd)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)

    best, best_result = False, 0
    start_time = time.perf_counter()
    for epoch in range(max_epoch):
        model.train()
        for batch_idx, (x, y, h, z) in enumerate(train_loader):
            # x = {k: v.cuda() for k, v in x.items()}
            # y = y.cuda()
            h = [item.cuda() for item in h]
            z = z.cuda()

            optimizer.zero_grad()
            outputs = model(**x, output_hidden_states=True, return_dict=True)
            if criterion_type in ("EAMKDLoss"):
                inters = masked_mean_pooling(outputs.hidden_states[-1], x["attention_mask"])
                loss = criterion(outputs.logits, inters, y, h, z)
            else:
                loss = criterion(outputs.logits, y, z)
            loss.backward()
            optimizer.step()

            if batch_idx % round(len(train_loader) * 0.15) == 0:
                train_log(optimizer, loss, batch_idx, epoch, max_epoch, train_loader)

        valid_loss, valid_result = _validate(model, criterion, criterion_type, valid_loader, metrics_average)
        valid_log(epoch, max_epoch, valid_loss, valid_result)
        if valid_result["f1_score"] > best_result:
            best_result = valid_result["f1_score"]
            best = True

        if epoch % save_interval == 0 or best:
            save_checkpoint(model, checkpoint_dir, epoch, best)
            best = False

    end_time = time.perf_counter()
    avg_time = (end_time - start_time) / max_epoch
    print(f"{criterion_type}: {avg_time}")


def evaluate(criterion, criterion_type, test_set, tokenizer_path, model_path, metrics_average, batch_size,
             if_correct=False):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).cuda()
    test_loader = DataLoader(MyDataset(test_set), batch_size=batch_size, shuffle=False, collate_fn=std_collate_fn(tokenizer))
    _, evaluate_result = _validate(model, criterion, criterion_type, test_loader, metrics_average, if_correct)
    return evaluate_result


def launch_train_evaluate(seed, criterion, criterion_type, dataset_name, if_train: bool, if_evaluate: bool,
                          if_sensitivity: str = "", if_ablation: str = ""):
    metrics_average, save_interval, max_epoch, train_bs, lr, wd, test_bs = get_config()

    model_path = "./files/pretrained/tinybert"
    tokenizer_path = "./files/pretrained/tinybert_tokenizer"
    checkpoint_dir = f"./files/checkpoint/student_{criterion_type}_{dataset_name}_{seed}{if_sensitivity}{if_ablation}"
    dataset = load_from_disk(f"./files/dataset/{dataset_name}_distilled_{seed}")
    if if_train:
        train_dataset = dataset["train"].filter(lambda example: example["text"].strip())
        train(criterion=criterion, train_set=train_dataset, valid_set=dataset["valid"], criterion_type=criterion_type,
              checkpoint_dir=checkpoint_dir, tokenizer_path=tokenizer_path, model_path=model_path,
              save_interval=save_interval, metrics_average=metrics_average, max_epoch=max_epoch,
              batch_size=train_bs, lr=lr, wd=wd)
    if if_evaluate:
        results = evaluate(model_path=f"{checkpoint_dir}/best_model", criterion=criterion, test_set=dataset["test"],
                           criterion_type=criterion_type, tokenizer_path=tokenizer_path, metrics_average=metrics_average,
                           batch_size=test_bs)
        print(results)
