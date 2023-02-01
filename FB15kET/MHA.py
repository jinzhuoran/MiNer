import torch
import torch.nn as nn


class CSRA(nn.Module):  # one basic block
    def __init__(self, T, lam):
        super(CSRA, self).__init__()
        self.T = T  # temperature
        self.lam = lam  # Lambda
        self.softmax = nn.Softmax(dim=1)

    def forward(self, score):

        base_logit = torch.mean(score, dim=1)

        if self.T == 99:  # max-pooling
            att_logit = torch.max(score, dim=1)[0]
        else:
            score_soft = self.softmax(score * self.T)
            att_logit = torch.sum(score * score_soft, dim=1)

        return base_logit + self.lam * att_logit


class MHA(nn.Module):  # multi-head attention
    temp_settings = {  # softmax temperature settings
        1: [3],
        2: [3, 99],
        3: [2, 4, 99],
        4: [2, 3, 4, 99],
        5: [2, 2.5, 3.5, 4.5, 99],
        6: [2, 3, 4, 5, 6, 99],
        7: [0.5, 2.5, 3.5, 4.5, 5.5, 6.5, 99],
        8: [0.5, 2, 3, 4, 5, 6, 7, 99]
    }

    def __init__(self, num_heads, lam, weight=False):
        super(MHA, self).__init__()
        self.num_heads = num_heads
        self.temp_list = self.temp_settings[num_heads]
        self.multi_head = nn.ModuleList([
            CSRA(self.temp_list[i], lam)
            for i in range(num_heads)
        ])
        self.weight = nn.Parameter(torch.ones(num_heads, 1))
        if weight:
            self.weight.requires_grad = True
        else:
            self.weight.requires_grad = False

    def forward(self, x):
        logit = 0.
        for head, weight in zip(self.multi_head, self.weight):
            logit += head(x) * weight
        return logit / self.num_heads
