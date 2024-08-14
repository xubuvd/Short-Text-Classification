import torch
from torch import nn
import math

def clones(module, N):
    '''Produce N identical modules
    '''
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def check_flag(d, key):
    '''check whether the dictionary `d` has `key` and `d[key]` is True
    '''
    return d.get(key) is not None and d.get(key)

class GELU(nn.Module):
    def forward(self, x):
        '''
        Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
        Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
        '''
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class Swish(nn.Module):
    def forward(self,x):
        '''
        Swish: a Self-Gated Activation Function (Ramachandran et al., https://arxiv.org/abs/1710.05941v1)
        where the SiLU was experimented with later.
        '''
        return x * torch.sigmoid(x)

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    #real layer-norm
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias

class MLPLayerNorm(nn.Module):
    def __init__(self,in_dim,out_dim,act=nn.GELU(),dropout=0.1):
        super(MLPLayerNorm, self).__init__()
        if act is not None:
            self.proj = nn.Sequential(
                nn.Linear(in_dim,out_dim,bias=True),
                act,
                nn.Dropout(dropout),
                nn.LayerNorm(out_dim)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(in_dim,out_dim,bias=True),
                nn.Dropout(dropout),
                nn.LayerNorm(out_dim)
            )
    def forward(self,x):
        return self.proj(x)

class GatedSoftLogit(nn.Module):
    def __init__(self,in_dim,out_dim,lastout_dim=1,act=nn.GELU(),dropout=0.1):
        super(GatedSoftLogit, self).__init__()
        self.proj = nn.Sequential(
            MLPLayerNorm(in_dim,out_dim,act,dropout),
            nn.Linear(out_dim,lastout_dim,bias=True)
        )
        #for m in self.modules():
        #    if isinstance(m, nn.Linear):
        #        nn.init.kaiming_uniform_(m.weight.data)
        #        if m.bias is not None: nn.init.constant_(m.bias.data, 0)
    def forward(self,input_x):
        return self.proj(input_x)

