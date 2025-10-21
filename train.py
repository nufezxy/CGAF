import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
import random
from loss import ContrastiveLoss
from dataloader import load_data
import time
from huxinxiloss import mi_loss_infoNCE_crossR


Dataname='Caltech-5V'
parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument("--learning_rate", default=0.0003)
parser.add_argument("--weight_decay", default=0.)
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
parser.add_argument('--lambda_epochs', type=int, default=10, metavar='N',
                        help='gradually increase the value of lambda from 0 to 1')
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--high_feature_dim", default=20)
parser.add_argument("--temperature", default=1)
parser.add_argument("--temperature_l", default=1.0)
parser.add_argument('--lam', default=0.1, type=float, help='weight for mutual information loss')#α
parser.add_argument("--lam2", default=1) #β
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.dataset == "Caltech-5V":
    args.con_epochs = 100
    args.lam = 0.1
    args.lam2 = 1
    seed = 100
if args.dataset == "NGs":
    args.con_epochs = 100
    args.lam = 0.2
    args.lam2 = 0.2
    seed = 40
if args.dataset == "Fashion":
    args.con_epochs = 100
    args.lam = 0.4
    args.lam2 = 1
    seed = 10
if args.dataset == "Cifar10":
    args.con_epochs = 20
    args.lam = 1
    args.lam2 = 1
    seed = 10
if args.dataset == "Synthetic3d":
    args.con_epochs = 100
    args.lam = 0.7
    args.lam2 = 0.8
    seed = 100
if args.dataset == "MNIST-USPS":
    args.con_epochs = 50
    args.lam = 0.9
    args.lam2 = 0.2
    seed = 10

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

seed=1
setup_seed(seed)

dataset, dims, view, data_size, class_num = load_data(args.dataset)
data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )

def pretrain(epoch):
    tot_loss = 0.
    criterion = torch.nn.MSELoss()
    for batch_idx, (xs, _, _) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs,_,_,_,_,_= model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(criterion(xs[v], xrs[v]))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss/len(data_loader)))


def contrastive_train(epoch, lam,lam2):
    tot_loss = 0.
    mse = torch.nn.MSELoss()
    for batch_idx, (xs, _, sample_idx) in enumerate(data_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
        optimizer.zero_grad()
        xrs, zs, rs, fused_rs, H, qs = model(xs)
        loss_list = []
        for v in range(view):
            loss_list.append(lam2 * contrastiveloss.forward_feature(H, fused_rs[v]))
            loss_list.append(mse(xs[v], xrs[v]))
        loss_list.append(lam * mi_loss_infoNCE_crossR(rs))
        loss = sum(loss_list)
        loss.backward()
        optimizer.step()
        tot_loss += loss.item()
    print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))




accs = []
nmis = []
purs = []
epochs = []
losses = []
if not os.path.exists('./models'):
    os.makedirs('./models')
T = 1


start_time = time.time()
#模型训练循环
for i in range(T):
    print("ROUND:{}".format(i+1))
    setup_seed(seed)
    epoch = args.pre_epochs + args.con_epochs
    model = Network(view, dims,class_num, args.feature_dim, args.high_feature_dim, device)
    print(model)
    model = model.to(device)
    state = model.state_dict()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    contrastiveloss = ContrastiveLoss(args.batch_size,class_num, args.temperature,args.temperature_l, device).to(device)
    best_acc, best_nmi, best_pur = 0, 0, 0

    epoch = 1
    while epoch <= args.pre_epochs:
        pretrain(epoch)
        epoch += 1

    while epoch <= args.pre_epochs + args.con_epochs:
        contrastive_train(epoch, lam=args.lam,lam2=args.lam2)
        acc, nmi, pur = valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
        if acc > best_acc:
            best_acc, best_nmi, best_pur = acc, nmi, pur
            state = model.state_dict()
            torch.save(state, './models/' + args.dataset + '.pth')
        epoch += 1

    print('The best clustering performace: ACC = {:.4f} NMI = {:.4f} PUR={:.4f}'.format(best_acc, best_nmi, best_pur))


end_time = time.time()  # 新增
print("Total training time: {:.2f} seconds".format(end_time - start_time))  # 新增