from typing import Literal
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn


def metrics_fn(y_true: list, y_pred: list,
               average: Literal["micro", "macro", "samples", "weighted", "binary", None] = 'macro',
               if_correct: bool = False):
    precision = precision_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    result = {'accuracy': round(accuracy * 100, 2), 'f1_score': round(f1 * 100, 2),
              'precision': round(precision * 100, 2), 'recall': round(recall * 100, 2)}
    if if_correct:
        correct = (np.asarray(y_pred) == np.asarray(y_true))
        correct = correct.astype(int).tolist()
        return result, correct
    return result

    # pre_0 = precision_score(y_true, y_pred, average=average, labels=[0])
    # rec_0 = recall_score(y_true, y_pred, average=average, labels=[0])
    # f1_0 = f1_score(y_true, y_pred, average=average, labels=[0])
    # pre_1 = precision_score(y_true, y_pred, average=average, labels=[1])
    # rec_1 = recall_score(y_true, y_pred, average=average, labels=[1])
    # f1_1 = f1_score(y_true, y_pred, average=average, labels=[1])
    # return {'accuracy': round(accuracy * 100, 2), 'f1_score': round(f1 * 100, 2),
    #         'precision': round(precision * 100, 2), 'recall': round(recall * 100, 2),
    #         'category_0_1': [[round(pre_0 * 100, 2), round(rec_0 * 100, 2), round(f1_0 * 100, 2)],
    #                          [round(pre_1 * 100, 2), round(rec_1 * 100, 2), round(f1_1 * 100, 2)]]}


def train_log(optimizer, loss, batch_idx, epoch, max_epoch, train_loader):
    print(f"training progress: [{epoch}/{max_epoch}, "
          f"{int(100 * round(batch_idx / len(train_loader), 1))}%], "
          f"lr: {optimizer.param_groups[0]['lr']:.6f}, "
          f"loss: {loss.item():.6f}")


def valid_log(epoch, max_epoch, loss, result):
    print(f"training epoch: [{epoch}/{max_epoch}], "
          f"valid loss: {loss:.6f}, "
          f"valid metrics: {result}")


def save_checkpoint(model, checkpoint_dir, epoch, save_best, if_peft: bool = False):
    filename = checkpoint_dir + f'/checkpoint-epoch-{epoch}'
    if if_peft:
        trainable_param_names = [
            name for name, param in model.named_parameters() if param.requires_grad
        ]
        trainable_state_dict = {
            name: param for name, param in model.state_dict().items() if name in trainable_param_names
        }
        torch.save(trainable_state_dict, f'{filename}.pth')
        print(f"Saved checkpoint: {filename}")
        if save_best:
            best_filename = checkpoint_dir + '/best_model'
            torch.save(trainable_state_dict, f'{best_filename}.pth')
            print(f"Saved current best: {best_filename}")
    else:
        model.save_pretrained(filename)
        print(f"Saved checkpoint: {filename}")
        if save_best:
            best_filename = checkpoint_dir + '/best_model'
            model.save_pretrained(best_filename)
            print(f"Saved current best: {best_filename}")


def masked_mean_pooling(hidden_states, attention_mask):
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    return sum_embeddings / sum_mask


def get_pred(model_name, tokenizer, model, text, if_attentions=False, layer=0, head=0):
    model.eval()
    with torch.no_grad():
        result = tokenizer(text, padding=True, truncation=True, max_length=192, return_tensors='pt')
        x = {k: v.cuda() for k, v in result.items()}
        outputs = model(**x, output_hidden_states=True, return_dict=True)
        if if_attentions:
            if model_name == "bart":
                # attentions = outputs.decoder_attentions[-1].mean(dim=1).squeeze()
                attentions = outputs.decoder_attentions[layer][0, head].squeeze()
            else:
                # attentions = outputs.attentions[-1].mean(dim=1).squeeze()
                attentions = outputs.attentions[layer][0, head].squeeze()
            tokens = tokenizer.convert_ids_to_tokens(result["input_ids"][0])
        if model_name == "bart":
            hidden_states = outputs.decoder_hidden_states[-1]
        else:
            hidden_states = outputs.hidden_states[-1]
        inters = masked_mean_pooling(hidden_states, x["attention_mask"])
        logits = outputs.logits
        prob = F.softmax(logits, dim=-1)
        pred = torch.argmax(prob, dim=1).squeeze()
        pred = pred.cpu().tolist()
    if if_attentions:
        return attentions, tokens
    else:
        return pred, prob, inters, logits


class Regressor(nn.Module):
    def __init__(self, tch_dim=768, std_dim=768, bn_dim=64):
        super().__init__()
        if tch_dim != std_dim:
            self.regressor = nn.Sequential(
                nn.Linear(tch_dim, bn_dim),
                nn.ReLU(),
                nn.Linear(bn_dim, std_dim)
            )
        else:
            self.regressor = nn.Identity()

    def forward(self, inters):
        h = self.regressor(inters)
        return h


class EAMKDLoss(nn.Module):
    def __init__(self, temp: float = 1, lamda: float = 0.95, gamma: float = 0.1, if_ablation: str = None):
        super().__init__()
        self.temp = temp
        self.lamda = lamda
        self.gamma = gamma

        self.tch1_regressor = Regressor()
        self.tch2_regressor = Regressor()
        self.tch3_regressor = Regressor()

        self.if_ablation = if_ablation

    def forward(self, logits, inters, y, h, z):
        tch1_inters = self.tch1_regressor(h[0])
        tch2_inters = self.tch2_regressor(h[1])
        tch3_inters = self.tch3_regressor(h[2])
        tch_inters = torch.stack([tch1_inters, tch2_inters, tch3_inters], dim=1)

        # attn_weights = self.get_attn_weights(logits, inters, z, tch_inters)  # (B, T)
        # fused_logits = torch.sum(z * attn_weights.unsqueeze(-1), dim=1)  # (B, C)
        # fused_inters = torch.sum(tch_inters * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        fused_logits, fused_inters = self.get_attn_weights(logits, inters, z, tch_inters)  # (B, T)

        hard_loss = F.cross_entropy(logits, y)
        soft_loss = F.kl_div(
            F.log_softmax(logits / self.temp, dim=-1),
            F.softmax(fused_logits / self.temp, dim=-1),
            reduction='batchmean'
        ) * (self.temp ** 2)
        inter_loss = F.mse_loss(inters, fused_inters)

        if self.if_ablation == "CE":
            total_loss = hard_loss
        elif self.if_ablation == "CE_MSE":
            total_loss = self.gamma * hard_loss + (1 - self.gamma) * inter_loss
        elif self.if_ablation == "CE_KL":
            total_loss = self.gamma * hard_loss + (1 - self.gamma) * soft_loss
        elif self.if_ablation == "MSE_KL":
            total_loss = self.lamda * soft_loss + (1 - self.lamda) * inter_loss
        else:
            total_loss = (self.gamma * hard_loss +
                          (1 - self.gamma) * (self.lamda * soft_loss + (1 - self.lamda) * inter_loss))
        return total_loss

    def get_attn_weights(self, std_logits, std_inters, tch_logits, tch_inters):
        std_logits = std_logits.unsqueeze(1).expand(-1, tch_logits.shape[1], -1)  # (B, T, C)
        std_inters = std_inters.unsqueeze(1).expand(-1, tch_logits.shape[1], -1)  # (B, T, D)

        kl_loss = F.kl_div(
            F.log_softmax(std_logits, dim=-1),
            F.softmax(tch_logits, dim=-1),
            reduction='none'
        ).sum(dim=-1)
        mse_loss = F.mse_loss(std_inters, tch_inters, reduction='none').sum(dim=-1)
        total_loss = kl_loss + mse_loss + 1e-8

        attn_scores = 1.0 / total_loss
        attn_weights = attn_scores / attn_scores.sum(dim=-1, keepdim=True)  # (B, T)
        # return attn_weights
        fused_logits = torch.sum(tch_logits * attn_weights.unsqueeze(-1), dim=1)  # (B, C)
        fused_inters = torch.sum(tch_inters * attn_weights.unsqueeze(-1), dim=1)  # (B, D)
        return fused_logits, fused_inters
