import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base model for this and many
    other models.
    """

    def __init__(self, encoder, src_embed, batch_size, d_model, n_class):
        super(TransformerEncoder, self).__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.linear = nn.Linear(d_model, n_class)

    def forward(self, src, src_mask):
        "Take in and process masked src and target sequences."
        enc = self.encoder(src, src_mask)
        # flat = enc.reshape(-1).unsqueeze(0)
        lin = self.linear(enc)
        result = F.softmax(lin, dim=1)
        return result

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)