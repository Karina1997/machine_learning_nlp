import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import random
import math
import time


class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 emb_dim, 
                 hid_dim, 
                 n_layers, 
                 kernel_size, 
                 dropout, 
                 device,
                 max_length = 100):
        super().__init__()

        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        assert kernel_size % 2 == 1, "Kernel size must be odd!"
        
        self.device = device
        
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        
        self.tok_embedding = nn.Embedding(input_dim, emb_dim)
        
        self.emb2hid = nn.Linear(emb_dim, hid_dim)
        self.hid2emb = nn.Linear(hid_dim, emb_dim)
        
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hid_dim, 
                                              out_channels = 2 * hid_dim, 
                                              kernel_size = kernel_size, 
                                              padding = (kernel_size - 1) // 2)
                                    for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)

        
    def forward(self, src):
        
        #src = [batch size, src len]
        
        batch_size = src.shape[1]
        src_len = src.shape[0]
        
        #create position tensor
        # pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [0, 1, 2, 3, ..., src len - 1]
        
        #pos = [batch size, src len]
        
        #embed tokens and positions
        tok_embedded = self.tok_embedding(src)
        # pos_embedded = self.pos_embedding(pos)
        
        #tok_embedded = pos_embedded = [batch size, src len, emb dim]
        
        #combine embeddings by elementwise summing
        embedded = self.dropout(tok_embedded)
        
        #embedded = [batch size, src len, emb dim]
        
        #pass embedded through linear layer to convert from emb dim to hid dim
        conv_input = self.emb2hid(embedded)
        
        #conv_input = [batch size, src len, hid dim]
        
        #permute for convolutional layer
        conv_input = conv_input.permute(0, 2, 1) 
        
        #conv_input = [batch size, hid dim, src len]
        
        #begin convolutional blocks...
        
        for i, conv in enumerate(self.convs):
        
            #pass through convolutional layer
            conved = conv(self.dropout(conv_input))

            #conved = [batch size, 2 * hid dim, src len]

            #pass through GLU activation function
            conved = F.glu(conved, dim = 1)

            #conved = [batch size, hid dim, src len]
            
            #apply residual connection
            conved = (conved + conv_input) * self.scale

            #conved = [batch size, hid dim, src len]
            
            #set conv_input to conved for next loop iteration
            conv_input = conved
        
        conved = conved.permute(0, 2, 1) 

        #conved = [src len, batch size, hid dim]


        #...end convolutional blocks
        
        #permute and convert back to emb dim
        # conved2 = self.hid2emb(conved.permute(0, 2, 1))
        
        #conved = [batch size, src len, emb dim]
        
        #elementwise sum output (conved) and input (embedded) to be used for attention
        # combined = (conved2 + embedded) * self.scale
        
        #combined = [batch size, src len, emb dim]
        pool = torch.max(conved, 0)
        pool = pool.values.view(1, batch_size, self.hid_dim)
        
        return pool, conved


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        
        self.attn = nn.Linear(enc_hid_dim, dec_hid_dim)
        self.softmax = nn.Softmax(dim = 2)
        
    def forward(self, hidden, encoder_outputs):

        # encoder_outputs = [sent_len?, batch_size, encoder_hid_dim]
        # hidden = [1, batch_size, decoder_hid_dim]

        output = self.attn(encoder_outputs)


        scores = torch.bmm(hidden.permute(1, 0, 2), output.permute(1, 2, 0))

        return self.softmax(scores)


class DecoderWithAttention(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, attention):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.attention = attention
        
        self.embedding = nn.Embedding(output_dim, emb_dim)

        
        self.rnn = self.rnn = nn.GRU(
            input_size=emb_dim + enc_hid_dim,
            hidden_size=dec_hid_dim,
            dropout=dropout
        )
        
        self.out = nn.Linear(
          in_features = dec_hid_dim + enc_hid_dim + emb_dim, 
          out_features=output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, encoder_outputs):

        input = input.unsqueeze(0)

        inputEmbeded = self.dropout(self.embedding(input))


        attn_weights = self.attention(hidden, encoder_outputs)
        attn_applied = torch.bmm(attn_weights, encoder_outputs.permute(1, 0, 2)).permute(1, 0, 2)

        # attn_applied = [1 , batch_size, dec_hid_dim]
        concatedEmbAttn = torch.cat((inputEmbeded, attn_applied), 2)

        output, hidden = self.rnn(concatedEmbAttn, hidden)

        concatedAttnOutput = torch.cat((concatedEmbAttn, output), 2)

        prediction = self.out(concatedAttnOutput)

        return prediction.squeeze(0), hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        # assert encoder.hid_dim == decoder.hid_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert encoder.n_layers == decoder.n_layers, \
        #     "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src sent len, batch size]
        #trg = [trg sent len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, conved = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, conved)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
