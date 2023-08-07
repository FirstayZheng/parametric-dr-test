import torch
import torch.nn as nn
from audtorch.metrics.functional import pearsonr
from utils.knn import get_geodesic_pairwise_distance


class UMAPLoss(nn.Module):
    def __init__(
        self,
        batch_size,
        negative_sample_rate,
        _a,
        _b,
        device,
        repulsion_strength=1.0,
        **kwargs,
    ):
        super(UMAPLoss, self).__init__()
        self.batch_size = batch_size
        self.negative_sample_rate = negative_sample_rate
        self._a = _a
        self._b = _b
        self.repulsion_strength = repulsion_strength
        self.device = device

    # temp fix
    def is_warmup(self):
        return False
    
    def convert_distance_to_probability(self, distances):
        return 1.0 / (1.0 + self._a * distances**(2 * self._b))

    def compute_cross_entropy(self,
                              probabilities_graph,
                              probabilities_distance,
                              EPS=1e-4):
        attraction_term = -probabilities_graph * torch.log(
            torch.clamp(probabilities_distance, min=EPS, max=1.0))
        repellant_term = (-(1.0 - probabilities_graph) * torch.log(
            torch.clamp(1.0 - probabilities_distance, min=EPS, max=1.0)) *
                          self.repulsion_strength)
        # balance the expected losses between atrraction and repel
        CE = attraction_term + repellant_term
        return attraction_term, repellant_term, CE

    def forward(self, embed_to_from, placeholder=None):
        embedding_to, embedding_from = torch.chunk(embed_to_from, 2, dim=1)
        embedding_neg_to = torch.repeat_interleave(embedding_to,
                                                   self.negative_sample_rate,
                                                   dim=0)
        repeat_neg = torch.repeat_interleave(embedding_from,
                                             self.negative_sample_rate,
                                             dim=0)

        repeat_neg_idx = torch.arange(0, repeat_neg.shape[0])
        idx = torch.randperm(repeat_neg_idx.shape[0])
        repeat_neg_idx = repeat_neg_idx[idx].view(repeat_neg_idx.size()).long()
        embedding_neg_from = repeat_neg[repeat_neg_idx]

        distance_embedding = torch.cat(
            ((torch.norm(embedding_to - embedding_from, dim=1)),
             torch.norm(embedding_neg_to - embedding_neg_from, dim=1)), 0)

        # convert probabilities to distances
        probabilities_distance = self.convert_distance_to_probability(
            distance_embedding)

        # set true probabilities based on negative sampling
        probabilities_graph = torch.cat(
            (torch.ones(self.batch_size),
             torch.zeros(self.batch_size * self.negative_sample_rate)),
            0).to(self.device, non_blocking=True)

        (attraction_loss, repellant_loss,
         ce_loss) = self.compute_cross_entropy(
             probabilities_graph,
             probabilities_distance,
         )
        return torch.mean(ce_loss)


class UMAPGlobalLoss(UMAPLoss):
    def __init__(
        self,
        global_loss,
        total_steps,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_loss = global_loss
        self.warmup = int(0.05 * total_steps)
        self.total_steps = total_steps
        self.batch_count = 0

        self.nonlinear_graph=[]
        self.nonlinear_graph=[]

    def is_warmup(self):
        return self.batch_count < self.warmup

    # 全局损失，即线性损失
    def _global_feature_loss(self, x_embeddings, to_x):
        embedding_to, embedding_from = torch.chunk(x_embeddings, 2, dim=1)
        # print(embedding_to.shape)
        
        to_x = torch.flatten(to_x, start_dim=1)
        
        num = to_x.shape[0]
        high_matrix = to_x.unsqueeze(0).repeat(num, 1, 1).to(self.device)
        # print(high_matrix.shape)
        low_matrix = embedding_to.unsqueeze(0).repeat(num, 1, 1).to(self.device)
        # print(low_matrix.shape)
        # calc MDS: calculate the distance of pairwise in high-dimensional space and low-dimentional space

        high_dis = torch.norm(high_matrix - high_matrix.transpose(0, 1), dim=-1)
        low_dis = torch.norm(low_matrix - low_matrix.transpose(0, 1), dim=-1)

        corr = torch.mean(
            pearsonr(
                high_dis.to(self.device),
                low_dis.to(self.device),
            )[:, 0])
        return -corr

    def forward(self, embed_to_from, placeholder=None):

        # self.to_x = placeholder
        self.batch_count += 1

        loss = super().forward(embed_to_from)
        loss_g = self._global_feature_loss(embed_to_from, placeholder).to(
            self.device)
        # print("loss:", loss, "  loss_g:", loss_g)
        # print("warmup: ", self.warmup)

        # 根据训练次数调整权重
        if self.batch_count >= self.warmup:
            if self.batch_count <= 0.9 * self.total_steps:
                beta = max(0, self.global_loss * (1 - self.batch_count / (self.total_steps - self.warmup * 4)) ** 2)
            else:
                beta = 0
            alpha = max(0, 1 - beta)
            # print("beta alpha 1:", beta, alpha)

        else:
            alpha = 0
            beta = self.global_loss

        if self.batch_count % 100 == 0:
            print("beta alpha :", beta, alpha)
            print("non l",loss,"l loss",loss_g)
        # print("non linear", loss, "l loss", loss_g)

        return alpha * loss + beta * loss_g
        # return loss_g


class ISOMAPLoss(UMAPGlobalLoss):
    def _global_feature_loss(self, x_embeddings, to_x):
        embedding_to, embedding_from = torch.chunk(x_embeddings, 2, dim=1)
        to_x = torch.flatten(to_x, start_dim=1)
        num = to_x.shape[0]
        # print(f"to_x.shape {to_x.shape}")
        # print(f"x_embeddings.shape {x_embeddings.shape}")
        high_dis = get_geodesic_pairwise_distance(data=to_x.cpu().numpy())
        high_dis = torch.tensor(high_dis).cuda()
        # print(f"high_dis.shape{high_dis.shape}")
        low_matrix = embedding_to.unsqueeze(0).repeat(num, 1, 1)
        low_dis = torch.norm(low_matrix - low_matrix.transpose(0, 1), dim=-1)
        # print(f"low_dis.shape{low_dis.shape}")
        # calc Pearson
        corr = torch.mean(
            pearsonr(
                high_dis,
                low_dis,
            )[:, 0])
        return -corr


# register the loss class
LOSS_MODE = {
    'noGlobalLoss': UMAPLoss,
    'GlobalLoss': UMAPGlobalLoss,
    'ISOMAPLoss': ISOMAPLoss,
}
