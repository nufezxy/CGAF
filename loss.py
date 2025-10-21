import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, class_num, temperature, temperature_l, device):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature = temperature
        self.temperature_l = temperature_l
        self.device = device
        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")


    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


    def forward_feature(self, h_i, h_j):
        N =self.batch_size
        similarity_matrix = torch.matmul(h_i, h_j.T) / self.temperature
        positives = torch.diag(similarity_matrix)
        mask = torch.ones((N, N)).to(self.device)
        mask = mask.fill_diagonal_(0)
        nominator = torch.exp(positives)
        denominator = (mask.bool()) * torch.exp(similarity_matrix)
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / N
        return loss


