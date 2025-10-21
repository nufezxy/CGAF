import torch
from network import Network
from metric import valid
import argparse
from dataloader import load_data


Dataname = 'NGs'
parser = argparse.ArgumentParser(description='test')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument("--feature_dim", default=64)
parser.add_argument("--hide_feature_dim", default=20)
parser.add_argument("--high_feature_dim", default=20, type=int, help="High feature dimension")
parser.add_argument("--pre_epochs", default=200)
parser.add_argument("--con_epochs", default=50)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset, dims, view, data_size, class_num = load_data(args.dataset)
epoch = args.pre_epochs + args.con_epochs
model = Network(view, dims,class_num, args.feature_dim, args.high_feature_dim, device)
model = model.to(device)

checkpoint = torch.load('./models/' + args.dataset + '.pth')
model.load_state_dict(checkpoint)


if args.dataset == "NGs":
	args.con_epochs = 100
	seed = 40

model.eval()
print("Dataset:{}".format(args.dataset))
print("Datasize:" + str(data_size))
print("Loading models...")
epoch = args.pre_epochs + args.con_epochs
acc, nmi, pur= valid(model, device, dataset, view, data_size, class_num, eval_h=False, epoch=epoch)
