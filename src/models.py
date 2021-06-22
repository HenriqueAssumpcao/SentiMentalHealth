import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


from .weight_norm import weight_norm as wn

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

