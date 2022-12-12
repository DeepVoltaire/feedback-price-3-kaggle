import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class FeedbackModel(nn.Module):
    def __init__(self, backbone, p_dropout=0.0):
        super(FeedbackModel, self).__init__()
        self.config = AutoConfig.from_pretrained(backbone, output_hidden_states=True)

        self.backbone = AutoModel.from_pretrained(backbone, config=self.config)
        self.dropout = nn.Dropout(p_dropout)
        self.linear = nn.Linear(
            in_features=self.config.hidden_size * 2,
            out_features=6,
        )

    def forward(self, inputs):
        outputs = self.backbone(**inputs)
        seq_x = outputs[0]
        apool = torch.mean(seq_x, 1)
        mpool, _ = torch.max(seq_x, 1)
        x = torch.cat((apool, mpool), 1)
        x = self.dropout(x)
        return self.linear(x)
