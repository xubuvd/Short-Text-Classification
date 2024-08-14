# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import random

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='random', type=str, help='random or pre_trained')
parser.add_argument('--dataset', default='', type=str, help='selected dataset')
parser.add_argument('--word', action='store_true', help='True for word, False for char')
parser.add_argument('--pad', default=10, type=int,help='sentence padding size')

if __name__ == '__main__':

    args = parser.parse_args()
    for arg in vars(args): print("{:<20}: {}".format(arg, getattr(args, arg)))

    # For reproducibility.
    random.seed(4257)
    np.random.seed(4257)
    torch.manual_seed(4257)
    torch.cuda.manual_seed_all(4257)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(args.dataset, embedding)
    print(config)
    config.pad_size = args.pad
 
    start_time = time.time()
    print("Loading data...")
    vocab, train_data, dev_data, test_data = build_dataset(config, args.word)
    train_iter = build_iterator(train_data, config)
    dev_iter = build_iterator(dev_data, config)
    test_iter = build_iterator(test_data, config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    config.n_vocab = len(vocab)
    model = x.Model(config).to(config.device)
    if model_name != 'Transformer':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)


