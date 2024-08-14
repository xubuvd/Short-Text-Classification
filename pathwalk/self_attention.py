import torch
import torch.nn as nn

class LinearNoActivationLayer(nn.Module):
    def __init__(self,in_size,out_size,dropout=0.1):
        super(LinearNoActivationLayer,self).__init__()
        
        self.linear = nn.Sequential(
            nn.Linear(in_size,out_size),
        )
    def forward(self,x):
        return self.linear(x)

class DecoderSelfAttention(nn.Module):
    def __init__(self,hidden_size=512,max_sequence_length=20,dropout=0.2,use_norm=False):
        super(DecoderSelfAttention,self).__init__()

        self.query_linear = LinearNoActivationLayer(hidden_size,hidden_size,dropout)
        self.key_linear = LinearNoActivationLayer(hidden_size,hidden_size,dropout)
        self.value_linear = LinearNoActivationLayer(hidden_size,hidden_size,dropout)
        self.register_buffer(
            name='attention_mask',
            tensor=torch.tril(
                torch.ones(max_sequence_length,max_sequence_length),
                diagonal=0
            ).unsqueeze(0)
        ) # (1,20,20)
        self.use_norm = use_norm
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self,x):
        # x,[32,20,512]
        mixed_query = self.query_linear(x)#[32,20,512]
        mixed_key = self.key_linear(x)
        mixed_value = self.value_linear(x)

        mixed_key = mixed_key.permute(0, 2, 1)#[32,512,20]
        scores = torch.matmul(mixed_query, mixed_key)#[32,20,20]
        scores = scores * self.attention_mask - 1e10 * (1 - self.attention_mask)
        attn_prob = nn.Softmax(dim=-1)(scores)#[32,20,20]
        if self.use_norm:
            attn_out = self.norm(mixed_value + torch.matmul(attn_prob, mixed_value))#[32,20,512]
        else:
            attn_out = mixed_value + torch.matmul(attn_prob, mixed_value)
        return attn_out

class EncoderAttention(nn.Module):
    def __init__(self,hidden_size=512,max_sequence_length=20,dropout=0.2):
        super(EncoderAttention,self).__init__()

        self.query_linear = LinearNoActivationLayer(hidden_size*2,hidden_size,dropout)
        self.key_linear = LinearNoActivationLayer(hidden_size*2,hidden_size,dropout)
        self.value_linear = LinearNoActivationLayer(hidden_size*2,hidden_size,dropout)
        self.register_buffer(
            name='attention_mask',
            tensor=torch.tril(
                torch.ones(max_sequence_length,max_sequence_length),
                diagonal=0
            ).unsqueeze(0)
        ) # (1,20,20)
        self.self_attn_linear = nn.Sequential(
            #nn.Linear(hidden_size, hidden_size),
            #GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        self.use_norm = False
        if self.use_norm:
            self.norm = nn.LayerNorm(hidden_size)

    def forward(self,x,mask_x):
        '''
        Input parameters:
            x:[32,20,1024]
            mask_x:[32,20]
        Return:
            summarized_vector:[32,512]
        '''
        mixed_query = self.query_linear(x)#[32,20,512]
        mixed_key = self.key_linear(x)
        mixed_value = self.value_linear(x)

        mixed_key = mixed_key.permute(0, 2, 1)#[32,512,20]
        scores = torch.matmul(mixed_query, mixed_key)#[32,20,20]
        scores = scores * self.attention_mask - 1e10 * (1 - self.attention_mask)
        attn_prob = nn.Softmax(dim=-1)(scores)#[32,20,20]
        attn_out = mixed_value + torch.matmul(attn_prob, mixed_value)#[32,20,512]

        self_attn_score = self.self_attn_linear(attn_out)#[32,20,1]
        self_attn_score = self_attn_score.masked_fill(mask_x.unsqueeze(-1) == 0, value=-9e10)
        self_attn_prob = nn.Softmax(dim=-2)(self_attn_score)#[32,20,1]
        summarized_vector = torch.matmul(self_attn_prob.transpose(-2, -1), attn_out).squeeze(1)#[32,1,512]->[32,512]
        if self.use_norm:
            summarized_vector = self.norm(summarized_vector)#[32,512]
        return summarized_vector

class DSTMemory():
    def __init__(self,config):
        super(DSTMemory,self).__init__()
        self.dialog_state = Parameter(
            nn.init.normal_(torch.empty(config["lstm_hidden_size"],config["lstm_hidden_size"]),std=0.02)
        )
        self.dialog_state_norm = nn.LayerNorm(config["lstm_hidden_size"])        
        self.weight = Parameter(nn.init.normal_(torch.empty(config["lstm_hidden_size"],1),std=0.02))
        self.bias = Parameter(torch.zeros(config["lstm_hidden_size"]))

        self.add = nn.Sequential(
            nn.Linear(config["lstm_hidden_size"],config["lstm_hidden_size"],bias=True),
        )
        self.batch_attn_linear = nn.Sequential(
            nn.LayerNorm(config["lstm_hidden_size"]),
            nn.Dropout(config["dropout"]),
            nn.Linear(config["lstm_hidden_size"],1)
        )
    def add(self,x):
        logit = self.batch_attn_linear(x)#[batch_size,1]
        prob = torch.softmax(logit,dim=0)#[batch_size,1]
        summarized_vector = torch.matmul(prob.transpose(-2, -1), x)#[1,lstm_hidden_size]
        delta = torch.addmm(self.bias,self.weight,summarized_vector)#[lstm_hidden_size,lstm_hidden_size]
        self.dialog_state = self.dialog_state + delta
        self.dialog_state = self.dialog_state_norm(self.dialog_state)
    def get(self,x):
        pass

class UnitSelfAttention(nn.Module):
    """This module perform self-attention on an utility
    to summarize it into a single vector."""

    def __init__(self, hidden_size,dropout=0.1):
        super(UnitSelfAttention, self).__init__()
        self.attn_linear = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, mask_x):
        """
        Arguments
        ---------
        x: torch.FloatTensor
        	The input tensor which is a sequence of tokens
        	Shape [batch_size, M, hidden_size]
        mask_x: torch.LongTensor
            The mask of the input x where 0 represents the <PAD> token
        	Shape [batch_size, M]
        Returns
        -------
        summarized_vector: torch.FloatTensor
            The summarized vector of the utility (the context vector for this utility)
            Shape [batch_size, hidden_size]
        """
        # shape [bs, M, 1]
        attn_weights = self.attn_linear(x)
        attn_weights = attn_weights.masked_fill(mask_x.unsqueeze(-1) == 0, value=-9e10)
        attn_weights = torch.softmax(attn_weights, dim=-2)

        # shape [bs, 1, hidden_size]
        summarized_vector = torch.matmul(attn_weights.transpose(-2, -1), x)
        summarized_vector = self.layer_norm(summarized_vector.squeeze(dim=-2))
        return summarized_vector

