import os
import sys
import torch
import torch.nn as nn

# ====================================================================
# [Config] Absolute path to model weights
# ====================================================================
MODEL_PATH = "/data/stu1/wrc3_pycharm_project/PromoDGDE_main/Predictor/wrc_class5_DNABERT2/DNABERT_2"

# Retain this constant for compatibility with training-script output
INPUT_CHANNELS = 128

# Add the path so bert_layers.py can be imported
sys.path.append(MODEL_PATH)

try:
    from bert_layers import BertModel
except ImportError:
    print(f"Error: Cannot find bert_layers.py in {MODEL_PATH}. Ensure the file exists and its internal imports use '.'.")
    raise


# ---------- Self-attention pooling ----------
class AttnPool1d(nn.Module):
    """
    Self-attention pooling:
        scores = w^T h_i
        alpha  = softmax(scores)
        h_pool = sum(alpha_i * h_i)
    Input:  (B, L, H)
    Output: (B, H)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, L, H)
        scores = self.attn(x).squeeze(-1)        # (B, L)
        weights = torch.softmax(scores, dim=-1)  # (B, L)
        pooled = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return pooled


# ==========================================
# Fine-tuned DNABERT2 + attention pooling + lightweight DNABert2_MLP head
# ==========================================
class DNABERT2_Hybrid_Model(nn.Module):

    def __init__(self,
                 num_classes: int = 5,
                 mlp_hidden_size: int = 256,
                 dropout_rate: float = 0.2,
                 use_last4: bool = True,
                 pool_type: str = "attn"):

        super().__init__()

        print(f">> [Model] Loading DNABERT-2 from: {MODEL_PATH} ...")
        self.bert = BertModel.from_pretrained(MODEL_PATH)

        self.use_last4 = use_last4
        self.pool_type = pool_type

        # Enable hidden-state output if supported by the config
        if hasattr(self.bert, "config") and hasattr(self.bert.config, "output_hidden_states"):
            self.bert.config.output_hidden_states = use_last4

        if hasattr(self.bert, "config") and hasattr(self.bert.config, "hidden_size"):
            self.bert_hidden = self.bert.config.hidden_size
        else:
            self.bert_hidden = 768  # Fallback

        # Pooling layer
        if self.pool_type == "attn":
            self.attn_pool = AttnPool1d(self.bert_hidden)
        else:
            self.attn_pool = None  # Use CLS

        # Lightweight DNABert2_MLP classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.bert_hidden),
            nn.Linear(self.bert_hidden, mlp_hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(mlp_hidden_size, num_classes),
        )

    # ---------- Extract token representations from BERT output ----------
    def _get_token_reps(self, outputs):

        hidden_states = None
        last_hidden_state = None

        if hasattr(outputs, "last_hidden_state"):
            last_hidden_state = outputs.last_hidden_state
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            last_hidden_state = outputs[0]

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden_states = outputs.hidden_states

        if self.use_last4 and hidden_states is not None and len(hidden_states) >= 4:
            last_4 = hidden_states[-4:]  # Tuple of four tensors, each (B, L, H)
            token_reps = torch.stack(last_4, dim=0).mean(dim=0)  # (B, L, H)
        else:
            if last_hidden_state is None:
                raise RuntimeError("Cannot parse BERT output. Check the return type of BertModel.")
            token_reps = last_hidden_state  # (B, L, H)

        return token_reps

    # ---------- Sequence-level pooling ----------
    def _pool(self, token_reps, attention_mask=None):

        if self.pool_type == "cls":
            pooled = token_reps[:, 0, :]  # (B, H)
        else:
            pooled = self.attn_pool(token_reps)  # (B, H)
        return pooled

    def forward(self, input_ids, attention_mask=None):
        # 1. Run DNABERT2
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=self.use_last4
        )

        # 2. Extract token-level representations
        token_reps = self._get_token_reps(outputs)  # (B, L, H)

        # 3. Pool into a sequence-level vector
        pooled = self._pool(token_reps, attention_mask=attention_mask)  # (B, H)

        # 4. Classification head
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits