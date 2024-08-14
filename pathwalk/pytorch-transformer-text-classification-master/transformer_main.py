import torch
import torch.nn as nn
import transformer
from data_load import load_train_data, next_batch
from transformer.misc import MultiHeadedAttention, PositionalEncoding, PositionwiseFeedForward, Embeddings
import argparse
from transformer.transformer import TransformerEncoder
from transformer.transformer_encoder import *


def make_model(src_vocab, N=6,
               d_model=512, d_ff=2048, h=8, dropout=0.1, batch_size=10, n_class=15):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = TransformerEncoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        batch_size,
        d_model,
        n_class
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def get_criterion(model, lr=0.005):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr)
    return criterion, optimizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--dim_model", type=int, default=128)
    parser.add_argument("--num_iterations", type=int, default=100000)
    parser.add_argument("--vocab_path", type=str, default="./corpora/vocab.txt")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'gpu')

    with open(args.vocab_path, "rt") as f:
        n_vocab = len(f.readlines())
    X, class_weights = load_train_data("./data")
    n_class = len(class_weights)
    model = make_model(n_vocab, d_model=args.dim_model, batch_size=args.batch_size, n_class=n_class)
    model = model.to(device)
    print(model)
    # epochs = 10
    for iter in range(args.num_iterations):
        x, y = next_batch(X, args.batch_size, args.dim_model)
    # batches = get_batches(in_text, out_text, 10, 200)
    # for x, y in batches:
        x = torch.tensor(x, dtype=torch.float, device=device)
        y = torch.tensor(y, dtype=torch.long, device=device)
        y = torch.squeeze(y)
        criterion, optimizer = get_criterion(model)
        optimizer.zero_grad()
        output = model(x, None)
        loss = criterion(output,y)
        if iter % 100 == 0:
            _, preds = torch.max(output, dim=1)
            print("Iteration {}: loss= {}, accuracy= {}".format(iter,loss.item(), float(torch.sum(preds==y).item()/args.batch_size)))
        loss.backward()
        optimizer.step()