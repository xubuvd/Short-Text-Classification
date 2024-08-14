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
print("pytorch version:{}".format(torch.__version__))

"""
Walk-of-Words model plus in-degree links for short text classification
"""
args = parser.parse_args()
config = yaml.safe_load(open(args.config_yml))

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
    num_class=len(class_list)
).to(device)
if len(args.gpu_ids) > 1 and torch.cuda.device_count() > 1: model = nn.DataParallel(model)

# Print config and args.
print(yaml.dump(config, default_flow_style=False))
for arg in vars(args):
    print("{:<20}: {}".format(arg, getattr(args, arg)))
print("parameters numbers:{}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

iterations = len(train_dataset) // config["solver"]["batch_size"] + 1
dev_iterations = len(dev_dataset) // config["solver"]["batch_size"] + 1
test_iterations = len(test_dataset) // config["solver"]["batch_size"] + 1

if args.load_pthpath != "":
    model_state_dict, optimizer_state_dict, save_params = load_checkpoint(args.load_pthpath)
    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)
    model.eval()
    test_acc, test_loss, test_report, test_confusion = evaluate(
        model,test_dataloader,test_iterations,class_list,device, is_test=True)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    #print("Confusion Matrix...")
    #print(test_confusion)

    dev_acc, dev_loss, dev_report, dev_confusion = evaluate(
        model,dev_dataloader,dev_iterations,class_list,device, is_test=True)
    msg = 'Dev Loss: {0:>5.2},  Dev Acc: {1:>6.2%}'
    print(msg.format(dev_loss, dev_acc))
    print("Precision, Recall and F1-Score...")
    print(dev_report)

    train_acc, train_loss, train_report, train_confusion = evaluate(
        model,train_dataloader,iterations,class_list,device, is_test=True)
    msg = 'Train Loss: {0:>5.2},  Train Acc: {1:>6.2%}'
    print(msg.format(train_loss, train_acc))
    print("Precision, Recall and F1-Score...")
    print(train_report)
else:
    print("load_pthpath is null.")

