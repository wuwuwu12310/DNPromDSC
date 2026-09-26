import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, matthews_corrcoef, classification_report
)
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
import warnings


from model_dnabert2_mlp import DNABERT2_Hybrid_Model

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f">> Random seed set to {seed}")



class DNASequenceDataset(Dataset):
    def __init__(self,
                 sequences,
                 labels,
                 tokenizer,
                 max_len,
                 kmer: int = 6,
                 do_kmer: bool = False):

        self.sequences = list(sequences)
        self.labels = list(labels)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.kmer = kmer
        self.do_kmer = do_kmer

    def __len__(self):
        return len(self.sequences)

    def _seq2kmer(self, seq: str) -> str:
        k = self.kmer
        seq = seq.upper()
        if len(seq) < k:
            return seq
        return " ".join(seq[i:i + k] for i in range(len(seq) - k + 1))

    def __getitem__(self, idx):
        seq = str(self.sequences[idx])
        label = int(self.labels[idx])

        if self.do_kmer:
            text = self._seq2kmer(seq)
        else:
            text = seq

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DNABERT2_Hybrid_Trainer:
    def __init__(self,
                 dataset_name='wrc_dnabert2_simple_probe',
                 num_classes=5,

                 kfold_root='/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Data/SC/wrc_processed_5fold_CGR/kfold5_class5',


                 save_root='/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_DNABERT2/results/results_dnabert2_mlp1_dataCGR',


                 model_path="/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_DNABERT2/DNABERT_2",


                 batch_size=256,
                 max_len=100,
                 weight_decay=1e-2,
                 patience=10,
                 max_epochs=100,
                 warmup_epochs=5,


                 kmer: int = 6,
                 do_kmer: bool = False,
                 ):

        self.dataset_name = dataset_name
        self.num_classes = num_classes
        self.kfold_root = kfold_root
        self.save_root = save_root
        self.model_path = model_path

        self.batch_size = batch_size
        self.max_len = max_len
        self.weight_decay = weight_decay
        self.patience = patience
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        self.kmer = kmer
        self.do_kmer = do_kmer

        os.makedirs(save_root, exist_ok=True)
        self.log_path = os.path.join(self.save_root, f'{self.dataset_name}_training_log.txt')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f">> Loading Tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.use_amp = (self.device.type == 'cuda')

        print(f">> Trainer init | batch_size={self.batch_size}, max_len={self.max_len}, "
              f"kmer={self.kmer}, do_kmer={self.do_kmer}")

    def log_print(self, f, text):
        print(text)
        f.write(text + "\n")
        f.flush()

    def load_fold_data(self, fold_idx):
        fold_dir = os.path.join(self.kfold_root, f'fold{fold_idx}')
        df_train = pd.read_csv(os.path.join(fold_dir, 'train.csv'))
        df_val = pd.read_csv(os.path.join(fold_dir, 'val.csv'))

        train_ds = DNASequenceDataset(
            df_train['seq'].tolist(),
            df_train['label'].tolist(),
            self.tokenizer,
            self.max_len,
            kmer=self.kmer,
            do_kmer=self.do_kmer
        )
        val_ds = DNASequenceDataset(
            df_val['seq'].tolist(),
            df_val['label'].tolist(),
            self.tokenizer,
            self.max_len,
            kmer=self.kmer,
            do_kmer=self.do_kmer
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        return (df_train, df_val), train_loader, val_loader

    def build_model(self):

        model = DNABERT2_Hybrid_Model(num_classes=self.num_classes).to(self.device)
        return model

    @torch.no_grad()
    def evaluate(self, model, loader):
        model.eval()
        preds, targets = [], []
        for batch in loader:
            input_ids = batch['input_ids'].to(self.device)
            mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            out = model(input_ids, mask)
            preds.extend(out.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

        y_true, y_pred = np.array(targets), np.array(preds)

        acc = accuracy_score(y_true, y_pred)
        f1_m = f1_score(y_true, y_pred, average='macro')
        f1_w = f1_score(y_true, y_pred, average='weighted')
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        return acc, f1_m, f1_w, prec, rec, mcc, y_true, y_pred

    def train_one_fold(self, fold_idx, log_f):
        self.log_print(log_f, f"\n{'=' * 20} Fold {fold_idx} Start {'=' * 20}")

        (_, _), train_dl, val_dl = self.load_fold_data(fold_idx)
        model = self.build_model()

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


        bert_params = []
        head_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith("bert."):
                bert_params.append(param)
            else:
                head_params.append(param)

        optimizer = torch.optim.AdamW(
            [

                {'params': bert_params, 'lr': 2e-5},

                {'params': head_params, 'lr': 1e-3},
            ],
            weight_decay=self.weight_decay
        )

        scaler = GradScaler(enabled=self.use_amp)

        steps_per_epoch = len(train_dl)
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=5 * steps_per_epoch, T_mult=2, eta_min=1e-6
        )

        best_f1 = 0.0
        patience_counter = 0
        best_stats = {}
        best_report = ""

        for ep in range(self.max_epochs):
            model.train()
            total_loss = 0.0


            if ep < self.warmup_epochs:
                factor = float(ep + 1) / float(self.warmup_epochs)
                optimizer.param_groups[0]['lr'] = 2e-5 * factor  # BERT
                optimizer.param_groups[1]['lr'] = 1e-3 * factor  # Head

            pbar = tqdm(train_dl, desc=f"[Fold {fold_idx}] Ep {ep + 1:03d}", leave=False)

            for batch in pbar:
                input_ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad(set_to_none=True)


                with autocast(enabled=self.use_amp):
                    logits = model(input_ids, mask)
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

                if ep >= self.warmup_epochs:
                    scheduler.step()

                total_loss += loss.item()

                lr_bert = optimizer.param_groups[0]['lr']
                lr_head = optimizer.param_groups[1]['lr']
                pbar.set_postfix(
                    loss=f"{loss.item():.4f}",
                    bert_lr=f"{lr_bert:.1e}",
                    head_lr=f"{lr_head:.1e}"
                )

            avg_loss = total_loss / len(train_dl)

            acc, f1_m, f1_w, prec, rec, mcc, y_true, y_pred = self.evaluate(model, val_dl)
            torch.cuda.empty_cache()

            log_line = (f"[Fold {fold_idx}] Ep {ep + 1:03d} | Loss={avg_loss:.4f} | "
                        f"Acc={acc:.4f} | F1_m={f1_m:.4f} | F1_w={f1_w:.4f} | MCC={mcc:.4f}")
            self.log_print(log_f, log_line)

            if f1_m > best_f1:
                best_f1 = f1_m
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(self.save_root, f"best_fold{fold_idx}.pth"))
                best_stats = {'acc': acc, 'f1_m': f1_m, 'f1_w': f1_w, 'prec': prec, 'rec': rec, 'mcc': mcc}
                best_report = classification_report(y_true, y_pred, digits=4)
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    self.log_print(log_f, f"Early Stopping at Ep {ep + 1}")
                    break

        self.log_print(log_f, f"\n>>> Report Fold {fold_idx}:")
        self.log_print(log_f, best_report)
        return best_stats

    def run(self):
        with open(self.log_path, 'w') as log_f:
            header = (
                f"{'=' * 60}\n"
                f"DNABERT2 MLP Classifier (last4 + attn)\n"
                f"Data      : {self.kfold_root}\n"
                f"Model Path: {self.model_path}\n"
                f"Batch Size: {self.batch_size}\n"
                f"Diff LR   : BERT(2e-5) / Head(1e-3)\n"
                f"k-mer     : k={self.kmer}, do_kmer={self.do_kmer}\n"
                f"{'=' * 60}"
            )
            self.log_print(log_f, header)

            final_metrics = {'acc': [], 'f1_m': [], 'f1_w': [], 'prec': [], 'rec': [], 'mcc': []}

            for fold in range(1, 6):
                stats = self.train_one_fold(fold, log_f)
                if stats:
                    for k, v in stats.items():
                        final_metrics[k].append(v)

            self.log_print(log_f, f"\n{'=' * 20} Final 5-Fold Summary {'=' * 20}")
            summary_content = ""
            for k in ['acc', 'f1_m', 'f1_w', 'prec', 'rec', 'mcc']:
                if final_metrics[k]:
                    mean, std = np.mean(final_metrics[k]), np.std(final_metrics[k])
                    line = f"{k.upper().ljust(10)}: {mean:.4f} ± {std:.4f}"
                    self.log_print(log_f, line)
                    summary_content += line + "\n"

            with open(os.path.join(self.save_root, "final_summary_stats.txt"), "w") as f:
                f.write(summary_content)
        print("Done.")


if __name__ == "__main__":
    seed_everything(42)
    trainer = DNABERT2_Hybrid_Trainer(

        kmer=6,
        do_kmer=False
    )
    trainer.run()

