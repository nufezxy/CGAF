import torch
import torch.nn as nn
from torch.nn.functional import normalize
from sklearn.cluster import KMeans

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class GateModule(nn.Module):
    def __init__(self, high_feature_dim, hidden_dim=128):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(2 * high_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, high_feature_dim),
            nn.Sigmoid()
        )
    def forward(self, r_view, r_others_fused):
        gate_input = torch.cat([r_view, r_others_fused], dim=1)
        g = self.gate_net(gate_input)
        return g


class SimpleDotProductAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query_layer = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, query, r_others):
        Q = self.query_layer(query)
        R_others = torch.stack(r_others, dim=1)
        scores = torch.bmm(R_others, Q.unsqueeze(-1)).squeeze(-1)
        alpha = self.softmax(scores)
        fused = torch.sum(alpha.unsqueeze(-1) * R_others, dim=1)
        return fused


class AttentionGateFusion(nn.Module):
    def __init__(self, view, high_feature_dim, device, hidden_dim=128):
        super().__init__()
        self.view = view
        self.attn_modules = nn.ModuleList([
            SimpleDotProductAttention(high_feature_dim).to(device) for _ in range(view)
        ])
        self.gate_modules = nn.ModuleList([
            GateModule(high_feature_dim, hidden_dim).to(device) for _ in range(view)
        ])
        self.others_compressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(high_feature_dim * (view - 1), hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, high_feature_dim)
            ).to(device) for _ in range(view)
        ])

    def forward(self, rs, H):
        fused = []
        for v in range(self.view):
            r_view = rs[v]
            r_others = [rs[i] for i in range(self.view) if i != v]
            r_others_concat = torch.cat(r_others, dim=1)
            r_others_fused = self.others_compressors[v](r_others_concat)
            fused_r = self.attn_modules[v](H, r_others)
            g = self.gate_modules[v](r_view, r_others_fused)
            fused.append((r_view + g * fused_r) / 2)

        return fused


class Network(nn.Module):
    def __init__(self, view, input_size, num_clusters, feature_dim, high_feature_dim, device):
        super().__init__()
        self.view = view
        self.num_clusters = num_clusters
        self.device = device

        self.encoders = nn.ModuleList([
            Encoder(input_size[v], feature_dim).to(device) for v in range(view)
        ])
        self.decoders = nn.ModuleList([
            Decoder(input_size[v], feature_dim).to(device) for v in range(view)
        ])

        self.common_information_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim)
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(high_feature_dim, num_clusters),
            nn.Softmax(dim=1)
        )
        self.feature_fusion_module = nn.Sequential(
            nn.Linear(view * feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, high_feature_dim)
        )

        self.fusion_module = AttentionGateFusion(view, high_feature_dim, device)

    def feature_fusion(self, zs, zs_gradient):
        input = torch.cat(zs, dim=1) if zs_gradient else torch.cat(zs, dim=1).detach()
        return normalize(self.feature_fusion_module(input), dim=1)

    def forward(self, xs, zs_gradient=True):
        rs, xrs, zs, qs = [], [], [], []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            xr = self.decoders[v](z)
            r = normalize(self.common_information_module(z), dim=1)
            q = self.label_contrastive_module(r)
            zs.append(z)
            xrs.append(xr)
            rs.append(r)
            qs.append(q)

        H = self.feature_fusion(zs, zs_gradient)
        fused_rs = self.fusion_module(rs, H)

        return xrs, zs, rs, fused_rs, H, qs
