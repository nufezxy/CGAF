import torch
import torch.nn.functional as F
import sys

def mi_loss_infoNCE_crossR(R_list, temperature=0.5):
    """
        temperature: 温度系数,不同数据集不同
        NGs:temperature=1;
        Fashion:temperature=0.5;
        MNIST-USPS:temperature=0.6;
        Caltech-5V:temperature=0.5;
        Synthetic3d:temperature=0.2;
        Cifar10:temperature=0.2;
    """
    m = len(R_list)
    N = R_list[0].size(0)
    device = R_list[0].device

    total_loss = 0.0
    count = 0

    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            Ri = F.normalize(R_list[i], dim=-1)
            Rj = F.normalize(R_list[j], dim=-1)

            sim_matrix = torch.matmul(Ri, Rj.t()) / temperature


            labels = torch.arange(N, device=device)

            loss_ij = F.cross_entropy(sim_matrix, labels)

            total_loss += loss_ij
            count += 1

    return total_loss / count



























































def compute_joint(x_out, x_tf_out):
    # 计算视图间的联合分布
    bn, k = x_out.size()
    assert (x_tf_out.size(0) == bn and x_tf_out.size(1) == k)

    p_i_j = x_out.unsqueeze(2) * x_tf_out.unsqueeze(1)  # bn, k, k
    p_i_j = p_i_j.sum(dim=0)  # k, k
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # 对称化
    p_i_j = p_i_j / p_i_j.sum()  # 归一化

    return p_i_j

def compute_feature_redundancy(x_out, x_tf_out, EPS=sys.float_info.epsilon):
    """特征间冗余互信息损失"""
    bn, k = x_out.size()
    H_cat = torch.cat([x_out, x_tf_out], dim=0)  # 合并两个视图的样本
    H_cat_norm = torch.nn.functional.normalize(H_cat, p=2, dim=1)

    P_feat_joint = torch.mm(H_cat_norm.T, H_cat_norm) / (2 * bn)

    # 非对角线部分
    mask = ~torch.eye(k, dtype=torch.bool, device=x_out.device)
    P_dd = P_feat_joint[mask].view(k, k - 1)
    P_d = torch.diag(P_feat_joint).view(k, 1)

    P_dd = torch.where(P_dd < EPS, torch.tensor([EPS], device=P_dd.device), P_dd)

    # 匹配维度的外积（也去除对角线）
    P_d_outer = P_d @ P_d.T  # [k, k]
    P_d_outer_no_diag = P_d_outer[mask].view(k, k - 1)

    loss_feature_redundancy = (P_dd * torch.log(P_dd / (P_d_outer_no_diag + EPS))).mean()
    return loss_feature_redundancy

def instance_contrastive_Loss(x_out, x_tf_out, lamb=10, EPS=sys.float_info.epsilon):
    """最大化视图一致性 + 最小化特征冗余"""
    _, k = x_out.size()
    p_i_j = compute_joint(x_out, x_tf_out)

    assert (p_i_j.size() == (k, k))
    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    # 视图一致性损失
    loss_consistency = - p_i_j * (torch.log(p_i_j) \
                                 - lamb * torch.log(p_j) \
                                 - lamb * torch.log(p_i))
    loss_consistency = loss_consistency.sum()


    # 主损失函数
    loss = loss_consistency/(k*k)
    return loss