import torch
import torch.nn as nn
import torch.nn.functional as F


class FNA(nn.Module):
    def __init__(self, eta):
        super(FNA, self).__init__()
        self.eta = eta
        self.loss = torch.nn.BCELoss(reduction='none')

    def forward(self, predict, label):
        loss = torch.nn.BCELoss(reduction='none')
        output = loss(predict, label)
        positive_loss = output * label
        negative_weight = predict.detach()
        negative_weight = self.eta * (negative_weight - negative_weight.pow(2)) * (1 - label)
        negative_loss = negative_weight * output
        return positive_loss.mean(), negative_loss.mean()


class TCR(nn.Module):
    def __init__(self, occurrence, num_examp, num_classes, lambd=3, omega=0.7, gama=0.96):
        super(TCR, self).__init__()
        self.occurrence = occurrence
        self.num_classes = num_classes
        self.USE_CUDA = torch.cuda.is_available()
        self.target = torch.zeros(num_examp, self.num_classes).cuda() if self.USE_CUDA else torch.zeros(num_examp,
                                                                                                        self.num_classes)
        self.omega = omega
        self.lambd = lambd
        self.gama = gama

    def forward(self, index, output, k):
        y_pred = F.softmax(output, dim=1)
        y_occ = F.softmax(torch.mm(self.occurrence, output.T).T, dim=1)
        y_pred_ = (1 - self.gama ** k) * y_pred.data.detach() + (self.gama ** k) * y_occ.data.detach()
        self.target[index] = self.omega * self.target[index] + (1 - self.omega) * (
                (y_pred_) / (y_pred_).sum(dim=1, keepdim=True))
        elr_reg = ((1 - (self.target[index] * y_pred).sum(dim=1)).log()).mean()
        final_loss = self.lambd * elr_reg
        return final_loss
