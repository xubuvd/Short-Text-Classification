# -*- coding:utf-8 -*-
import os
import sys
import numpy as np
import random
import torch
import pickle
import argparse
import itertools
import networkx as nx
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from sklearn import metrics
from bisect import bisect
from dataset import PathWalkDataset
from torch.nn import functional as F

from util import evaluate,test
from model import PathWalkModel
from checkpointing import CheckpointManager, load_checkpoint

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# For reproducibility.
random.seed(4257)
np.random.seed(4257)
torch.manual_seed(4257)
torch.cuda.manual_seed_all(4257)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

#Parameters
parser = argparse.ArgumentParser()
parser.add_argument(
    "--config-yml",
    default="configs/pathwalk.yml",
    help="Path to a config file listing reader, model and solver parameters.",
)
parser.add_argument(
    "--cpu-workers",
    type=int,
    default=4,
    help="Number of CPU workers for dataloader.",
)
parser.add_argument(
    "--load_pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint.",
)
parser.add_argument(
    "--validate",
    action="store_true",
    help="Whether to validate on val split after every epoch.",
)
parser.add_argument(
    "--overfit",
    action="store_true",
    help="Overfit model on 5 examples, meant for debugging.",
)
parser.add_argument(
    "--gpu-ids",
    #nargs="+",
    #type=int,
    default="1",
    help="List of ids of GPUs to use.",
)
parser.add_argument(
    "--in-memory",
    action="store_true",
    help="Load the whole dataset and pre-extracted image features in memory. "
    "Use only in presence of large RAM, at least few tens of GBs.",
)

parser.add_argument_group("Checkpointing related arguments")
parser.add_argument(
    "--save-dirpath",
    default="",
    help="Path of directory to create checkpoint directory and save "
    "checkpoints.",
)
parser.add_argument(
    "--load-pthpath",
    default="",
    help="To continue training, path to .pth file of saved checkpoint.",
)
parser.add_argument(
    "--select_indegree_num",
    default=-1,
    type=int,
    help="number of skip-path based neighbors"
)
parser.add_argument(
    "--embedding_size",
    default=-1,
    type=int,
    help=""
)
parser.add_argument(
    "--max_sequence_length",
    default=-1,
    type=int,
    help=""
)


print("pytorch version:{}".format(torch.__version__))

"""
Walk-of-Words model plus in-degree links for short text classification
"""
args = parser.parse_args()
config = yaml.safe_load(open(args.config_yml))

#config["dataset"]["select_indegree_num"] = args.select_indegree_num
#print("select_indegree_num: {}".format(config["dataset"]["select_indegree_num"]))


#config["model"]["embedding_size"] = args.embedding_size
#print("model.embedding_size: {}".format(args.embedding_size))

#config["model"]["max_sequence_length"] = args.max_sequence_length
#print("model.max_sequence_length: {}".format(args.max_sequence_length))

## Print config and args
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("args {:<20}: {}".format(arg, getattr(args, arg)))

G_networkx = nx.read_gpickle(config["dataset"]["graph_gpickle"])
with open(config["dataset"]["edge2id_pkl"],'rb') as fb: Edge2id_dict = pickle.load(fb)

class_list = [x.strip() for x in open(config["dataset"]["class_list"], encoding='utf-8').readlines()]

train_dataset = PathWalkDataset(config=config["dataset"],
    input_file=config["dataset"]["train_file_path"],
    graph=G_networkx,
    edge2id=Edge2id_dict,
    use_word=config["dataset"]["use_word"],
    loop_topology=True,
    overfit=args.overfit)
point_size = train_dataset.edge_size + train_dataset.vocab_size
print("node size:{}\tedge size:{}".format(train_dataset.vocab_size,train_dataset.edge_size))
train_dataloader = DataLoader(
    train_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
    pin_memory=True,
    drop_last=True)

dev_dataset = PathWalkDataset(config=config["dataset"],
    input_file=config["dataset"]["dev_file_path"],
    graph=G_networkx,
    edge2id=Edge2id_dict,
    use_word=config["dataset"]["use_word"],
    loop_topology=True)
dev_dataloader = DataLoader(
    dev_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
    pin_memory=True,
    drop_last=True)

test_dataset = PathWalkDataset(config=config["dataset"],
    input_file=config["dataset"]["test_file_path"],
    graph=G_networkx,
    edge2id=Edge2id_dict,
    use_word=config["dataset"]["use_word"],
    loop_topology=True)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=config["solver"]["batch_size"],
    num_workers=args.cpu_workers,
    shuffle=True,
    pin_memory=True,
    drop_last=False)

if isinstance(args.gpu_ids, int):
    args.gpu_ids = [args.gpu_ids]
else:
    args.gpu_ids = [int(item) for item in args.gpu_ids.strip().split(",")]

device = (
    torch.device("cuda")
    if args.gpu_ids[0] >= 0
    else torch.device("cpu")
)
print("device:{}".format(device))
model = PathWalkModel(
    config=config["model"],
    node_size=point_size,
    vocabulary=train_dataset.vocabulary,
    num_class=len(class_list),
).to(device)
if len(args.gpu_ids) > 1 and torch.cuda.device_count() > 1: model = nn.DataParallel(model)

if False == os.path.exists(args.save_dirpath): os.makedirs(args.save_dirpath)

cmd="cp -rf ./start.sh *.py  {}".format(args.save_dirpath);os.system(cmd)
cmd="cp -rf ./configs {}".format(args.save_dirpath);os.system(cmd)
print("parameters numbers:{}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

iterations = len(train_dataset) // config["solver"]["batch_size"] + 1
dev_iterations = len(dev_dataset) // config["solver"]["batch_size"] + 1
test_iterations = len(test_dataset) // config["solver"]["batch_size"] + 1

def lr_lambda_fun(current_iteration: int) -> float:
    current_epoch = float(current_iteration) / iterations
    if current_epoch <= config["solver"]["warmup_epochs"]:
        alpha = current_epoch / float(config["solver"]["warmup_epochs"])
        return config["solver"]["warmup_factor"] * (1.0 - alpha) + alpha
    else:
        idx = bisect(config["solver"]["lr_milestones"], current_epoch)
        return pow(config["solver"]["lr_gamma"], idx)

optimizer = optim.Adamax(model.parameters(), lr=config["solver"]["initial_lr"])
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda_fun)

checkpoint_manager = CheckpointManager(
    model, optimizer, args.save_dirpath, config=config
)

save_params = {}
if args.load_pthpath == "":
    start_epoch = 0
    bestACC = 0.0
    lRate = config["solver"]["initial_lr"]
else:
    start_epoch = int(args.load_pthpath.split("_")[-1][:-4])
    model_state_dict, optimizer_state_dict, save_params = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    optimizer.load_state_dict(optimizer_state_dict)
    bestACC = save_params["bestACC"]
    lRate = save_params["lRate"]
    print("Loaded model from {}, bestMRR:{}, lRate:{}".format(args.load_pthpath,bestMRR,lRate))
global_iteration_step = start_epoch * iterations
no_increase_epochs = 0

for epoch in range(start_epoch, config["solver"]["num_epochs"]):
    model.train()
    print(f"\nTraining for epoch {epoch+1}:")
    loss_total = 0
    combined_dataloader = itertools.chain(train_dataloader)
    for i, batch in tqdm(enumerate(combined_dataloader),total=iterations):
        optimizer.zero_grad()
        for key in batch: batch[key] = batch[key].to(device)
        y = batch["y"]#[batch_size]
        y_hat = model(batch)#[batch_size,num_class]
        loss = F.cross_entropy(y_hat, y)
        loss_total += loss
        loss.backward()
        torch.nn.utils.clip_grad_value_(parameters=model.parameters(),clip_value=1.5)
        optimizer.step()

        global_iteration_step += 1
        scheduler.step()
        if i % 50 == 0:
            print("epoch:{}\tbatch_loss:{}".format(epoch+1,loss))
    train_loss_epoch = loss_total / iterations
    if args.validate:
        print(f"\nValidation after epoch {epoch+1}:")
        acc,dev_loss_epoch = evaluate(model,dev_dataloader,dev_iterations,class_list,device)
        if acc > bestACC:
            save_params["bestACC"] = acc
            save_params["lRate"] = lRate
            checkpoint_manager.step(params=save_params,epoch=epoch+1)
            bestACC = acc
            no_increase_epochs = 0
        else:
            no_increase_epochs += 1
        print("\nepoch:{}\tAcc:{}\ttrain_loss:{}\tdev_loss:{}\tlr:{}".format(
            epoch+1,acc,train_loss_epoch,dev_loss_epoch,optimizer.param_groups[0]["lr"]))
    if no_increase_epochs > config["solver"]["stopping_epochs"]:
        print("Early Stopping at epoch {}, as the acc does not increase in {} consecutive epochs".format(
            epoch+1,config["solver"]["stopping_epochs"]))
        break
test(checkpoint_manager.prev_savePath,model,test_dataloader,test_iterations,class_list,device)

