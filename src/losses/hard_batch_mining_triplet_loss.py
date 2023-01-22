import torch
import numpy as np
from utils.utils import euclidean_squared_distance
class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.
    
    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3,device=None):
        super(HardBatchMiningTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)
        self.device=device if device!=None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        n = inputs.size(0)

        # Compute the pairwise euclidean distance between all n feature vectors.
        distance_matrix = euclidean_squared_distance(inputs,inputs).clamp(min=1e-12).sqrt()

        # For each sample (image), find the hardest positive and hardest negative sample.
        # The targets are a vector that encode the class label for each of the n samples.
        # Pairs of samples with the SAME class can form a positive sample.
        # Pairs of samples with a DIFFERENT class can form a negative sample.
        distance_positive_pairs, distance_negative_pairs = [], []
        for i in range(n):
            class_i = targets[i]
            distances = distance_matrix[i,:]

            pos_dist = []
            neg_dist = []
            for j in range(n):
                if(j == i):
                    continue
                if(targets[j] == class_i):
                  pos_dist.append(distances[j].item())
                else:
                  neg_dist.append(distances[j].item())

            distance_positive_pairs.append(np.max(pos_dist))
            distance_negative_pairs.append(np.min(neg_dist))

        # Convert the created lists into 1D pytorch tensors
        distance_positive_pairs = torch.as_tensor(distance_positive_pairs)
        distance_positive_pairs = distance_positive_pairs.to(self.device)
        distance_negative_pairs = torch.as_tensor(distance_negative_pairs)
        distance_negative_pairs = distance_negative_pairs.to(self.device)

        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        y = torch.ones_like(distance_negative_pairs)
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)