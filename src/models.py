import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import attention as attn
from .weight_norm import weight_norm as wn

RNNS = ['LSTM', 'GRU']


class MyRNN(nn.Module):
  def __init__(self, params, device, rnn_type='GRU'):
    super(MyRNN, self).__init__()

    input_size  = params['input_size']
    hidden_size = params['hidden_size']
    output_size = params['output_size']

    bidirectional = params['bidirectional']

    dropout_rnn = params['dropout_enc']
    #dropout_out = params['dropout_out']

    self.bidirectional = bidirectional
    self.hidden_size = hidden_size
    self.device = device
    
    assert rnn_type in RNNS, 'Use one of the following: {}'.format(str(RNNS))
    rnn_cell = getattr(nn, rnn_type) # fetch constructor from torch.nn, cleaner than if
    self.rnn = rnn_cell(input_size, hidden_size, num_layers, 
                        dropout=dropout_rnn, batch_first=True, bidirectional=False)

  def forward(self, input):
    b, t, _ = input.size()
    k = self.hidden_size
    device = self.device
    outputs = torch.zeros(b,t,k).to(device)
    hiddens = torch.zeros(t,b,k).to(device)
    rev_outputs = torch.zeros(b,t,k).to(device)
    rev_hiddens = torch.zeros(t,b,k).to(device)

    hidden = None
    rev_hidden = None
    for tx in range(t):
      output, hidden = self.rnn(input[:,tx,:].unsqueeze(1), hidden)
      outputs[:,tx,:] = output.squeeze(1)
      hiddens[tx,:,:] = hidden[-1,:,:].unsqueeze(0)
      if self.bidirectional:
        rev_tx = -(tx+1)
        rev_output, rev_hidden = self.rnn(input[:,rev_tx,:].unsqueeze(1), rev_hidden)
        rev_outputs[:,rev_tx,:] = rev_output.squeeze(1)
        rev_hiddens[rev_tx,:,:] = rev_hidden[-1,:,:].unsqueeze(0)

    if self.bidirectional:
      outputs = torch.cat( (outputs, rev_outputs), dim=2 )
      hiddens = torch.cat( (hiddens, rev_hiddens), dim=2 )

    return outputs, hiddens

class SimpleEncoder(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, dropout=0.):
    super(SimpleEncoder, self).__init__()
    self.embedding_dim = embedding_dim
    self.hidden_dim = hidden_dim
    self.W = nn.Parameter(torch.Tensor(embedding_dim,hidden_dim))
    self.do = nn.Dropout(dropout)

  def forward(self, input, hidden=None):
    output = torch.matmul(input, self.W)
    output = self.do(output)
    output = F.relu(output)
    return output

class Classifier(nn.Module):
  def __init__(self, encoder, params):
    super(Classifier, self).__init__()
    self.encoder = encoder

    attention_type = params['attention_type']

    query_size  = params['query_size']
    query_type  = params['query_type']
    key_size    = params['key_size']
    output_size = params['output_size']
    dropout_att = params['dropout_att']
    dropout_out = params['dropout_out']
    weight_norm = params['weight_norm'] if 'weight_norm' in params else False
    num_heads   = params['num_heads']   if 'num_heads'   in params else -1


    if attention_type == 'OrderAttention':
      self.attention = attn.OrderAttention(dropout=dropout_att, causal=False)
    elif attention_type == 'MultiHeadAttention':
      self.attention = attn.MultiHeadAttention(query_size, query_size, num_heads=num_heads, dropout=dropout_att)
    elif attention_type == 'MultiHeadAttentionV2':
      self.attention = attn.MultiHeadAttentionV2(query_size, query_size, num_heads=num_heads, dropout=dropout_att)
    elif attention_type == 'AttentionLayer':
      self.attention = attn.AttentionLayer(query_size, key_size, batch_first=True,
                                dropout=dropout_att, output_transform=False, query_transform=False)
    else:
      raise ValuError('NO ATTENTION {} FOUND!'.format(attention_type))

    if query_size != key_size:
        wn_func = wn if weight_norm else lambda x: x
        self.linear_q = wn_func(nn.Linear(query_size, key_size))

    self.linear_out = nn.Linear(query_size+key_size, output_size)
    self.query_type = query_type
    self.m = nn.Tanh()

  def forward(self, input, seq_len):

    # use encoder to obtain query, keys and values
    outputs = self.encoder(input)
    if isinstance(outputs, tuple): # RNN
      outputs, hiddens = outputs
      if isinstance(hiddens, tuple): # LSTM
        hiddens = hiddens[1] # take the cell state
      hiddens = hiddens.permute(1,0,2) #[b, t*num_layers, k*(1+1*bidirectional)]
      hidden  = hiddens[:,0]
    else: # when encoder is SimpleEncoder
      hidden = outputs[:,0,:]

    # compute mask if there is set_mask or set_mask_k
    set_mask = getattr(self.attention, "set_mask", None)
    set_mask = getattr(self.attention, "set_mask_k", set_mask)
    if set_mask is not None:
      b   = input.size(0)
      t_k = hiddens.size(1)
      tensor_arange = torch.arange(0,t_k, device=seq_len.device).reshape(1,-1).expand([b,t_k])
      tensor_seqlen = seq_len.reshape(-1,1).expand([b,t_k])
      mask = tensor_arange >= tensor_seqlen
      set_mask(mask)

    # check what should be used as query
    if query_type == 'hidden':
      input_tensor = hidden
    elif query_type == 'output':
      input_tensor = outputs[:,0,:]
    else:
      input_tensor = input[:,0,:]

    # make sure input_tensor has the right number of dimensions
    if input_tensor.dim() == 2:
      input_tensor = input_tensor.unsqueeze(1)

    # if query_size and keys_size are different, do linear transformation on query
    if input_tensor.size(2) != outputs.size(2):
      input_tensor = self.linear_q(input_tensor)

    if isinstance(self.attention,attn.OrderAttention):
      attn_output_weights = self.attention(input_tensor, hiddens)
      attn_output = torch.bmm(attn_output_weights, hiddens)

    elif isinstance(self.attention,attn.AttentionLayer) or \
    isinstance(self.attention,attn.SDPAttention) or \
    isinstance(self.attention,attn.MultiHeadAttention) or \
    isinstance(self.attention,attn.MultiHeadAttentionV2):
      attn_output, attn_output_weights = self.attention(input_tensor, hiddens, hiddens)

    output = self.linear_out(torch.cat([input_tensor, attn_output], 2))
    output = self.m(output.squeeze(2))     

    return output, attn_output_weights

def _generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

class GRUSentiment(nn.Module):
    def __init__(self, params):
        
        super().__init__()
                
        input_size = params['input_size']
        hidden_size = params['hidden_size']
        num_layers = params['num_layers']
        output_size = params['output_size']
        bidirectional = params['bidirectional']
        dropout_rnn = params['dropout_rnn']
        dropout_out = params['dropout_out']

        self.fc = nn.Linear(input_size-2, hidden_size-2)
        
        self.rnn = nn.GRU(hidden_size, 
                          hidden_size//2,
                          num_layers,
                          batch_first = True,
                          bidirectional = bidirectional,
                          dropout = 0 if num_layers < 2 else dropout_rnn)
        
        self.out = nn.Linear(hidden_size+1 if bidirectional else hidden_size//2+1, output_size)
        self.do = nn.Dropout(dropout_out)
        self.relu = nn.ReLU()

        self.m = nn.Sigmoid()

    def forward(self, input, src_len):
        import pdb
        #text = [batch size, sent len]
                
        # with torch.no_grad():
        #     embedded = self.bert(text)[0]
                
        #embedded = [batch size, sent len, emb dim]
        embedded = self.fc(input[:,:,:-2])
        embedded = torch.cat([embedded,input[:,:,-2:]], axis=2)
        outputs, hidden = self.rnn(embedded)
        
        #hidden = [n layers * n directions, batch size, emb dim]
        
        if self.rnn.bidirectional:
            hidden = self.do(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            index = (src_len-1).reshape([-1,1,1]).repeat([1,1,outputs.size()[2]])
            hidden = torch.gather(outputs,1,index).squeeze(1)
            hidden = self.do(hidden)
                
        #hidden = [batch size, hid dim]

        #raw_output = [batch size, out dim]
        hidden = torch.cat([hidden,input[:,:,-2].mean(axis=1).unsqueeze(1)],axis=1)
        output = self.out(hidden)
        # output = self.m(output)

        #hidden[:,:,0].index_select(dim=0,index=src_len-1).diag()
        
        #output = [batch size, out dim]
        
        return output, None

class GRUSentiment2(nn.Module):
    def __init__(self, params):
        
        super().__init__()
                
        input_size    = params['input_size']
        hidden_size   = params['hidden_size']
        num_layers    = params['num_layers']
        output_size   = params['output_size']
        bidirectional = params['bidirectional']
        dropout_rnn   = params['dropout_rnn']
        dropout_out   = params['dropout_out']
        if 'uses_two_series_as_input' in params:
          uses_two_series_as_input = params['uses_two_series_as_input']
        else:
          uses_two_series_as_input = False

        self.rnn1 = nn.GRU(input_size, 
                            hidden_size,
                            num_layers,
                            batch_first = True,
                            bidirectional = bidirectional,
                            dropout = 0 if num_layers < 2 else dropout_rnn
                     )
        if uses_two_series_as_input:
            self.rnn2 = nn.GRU(input_size, 
                                hidden_size,
                                num_layers,
                                batch_first = True,
                                bidirectional = bidirectional,
                                dropout = 0 if num_layers < 2 else dropout_rnn
                        )
        else:
            self.rnn2 = None
        

        embedded_size = hidden_size * (1+1*uses_two_series_as_input) * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(embedded_size, 1)
        # self.fc1 = nn.Linear(embedded_size, embedded_size//2)
        # self.fc2 = nn.Linear(embedded_size//2,output_size)

        self.relu = nn.ReLU()
        self.do   = nn.Dropout(dropout_out)
        self.m = nn.Sigmoid()

        self.uses_two_series_as_input = uses_two_series_as_input

    def forward(self, x1, len_x1, x2=None, len_x2=None):
        hiddens = []
        for rnn, x, len_x in zip([self.rnn1, self.rnn2],[x1,x2],[len_x1,len_x2]):
          #x = [batch size, sent len, emb dim]
          outputs, hidden = rnn(x)
          #hidden = [n layers * n directions, batch size, emb dim]

          if rnn.bidirectional:
              hidden = self.do(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
          else:
              index = (len_x-1).reshape([-1,1,1]).repeat([1,1,outputs.size()[2]])
              hidden = torch.gather(outputs,1,index).squeeze(1)
              hidden = self.do(hidden)

          #hidden = [batch size, hid dim]
          hiddens.append(hidden)
          if x2 is None:
            break

        embedded = torch.cat(hiddens,dim=1)
        output = self.fc1(embedded)
        # output = self.do(output)
        # output = self.relu(output)

        # output = self.fc2(output)
        # output = self.do(output)
        output = self.m(output)      
        #output = [batch size, out dim]
        
        return output, None

class TransformerDecoderLayer(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)
    """

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            tgt: the sequence to the decoder layer (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class TransformerDecoder(nn.Module):
    r"""TransformerDecoder is a stack of N decoder layers
    Args:
        decoder_layer: an instance of the TransformerDecoderLayer() class (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = transformer_decoder(tgt, memory)
    """
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.linear = nn.Linear(770, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, tgt, memory, tgt_mask=None,
                memory_mask=None, tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        # type: (Tensor, Tensor, Optional[Tensor], Optional[Tensor], Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the inputs (and mask) through the decoder layer in turn.
        Args:
            tgt: the sequence to the decoder (required).
            memory: the sequence from the last layer of the encoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            memory_mask: the mask for the memory sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            memory_key_padding_mask: the mask for the memory keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)
        output = self.linear(output)
        output = self.sigmoid(output)

        return output