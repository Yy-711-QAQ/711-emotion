import torch
from torch import nn
from transformers import AutoModel

class MME2E_T(nn.Module):
    def __init__(self, feature_dim, num_classes=4, size='base'):
        super(MME2E_T, self).__init__()

        # 中文模型
        self.bert = AutoModel.from_pretrained("./pretrained/bert-base-chinese")

        hidden_size = self.bert.config.hidden_size

        self.text_feature_affine = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, feature_dim)
        )

    def forward(self, text, get_cls=False):
        outputs = self.bert(**text)
        last_hidden_state = outputs[0]


        if get_cls:
            cls_feature = last_hidden_state[:, 0]
            return cls_feature

        text_features = self.text_feature_affine(last_hidden_state).sum(1)
        return text_features
