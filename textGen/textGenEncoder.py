import torch
import json
import torch.nn as nn
from textGen import TextTransformer

# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the values from the configuration file
src_vocab_size = config['src_vocab_size']
embed_size = config['embed_size']
num_layers = config['num_layers']
num_heads = config['num_heads']
device = config['device']
forward_expansion = config['forward_expansion']
dropout = config['dropout']
max_length = config['max_length']

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length,):
        super(Encoder, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TextTransformer(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        # In the Encoder the query, key, value are all the same, it's in the decoder this will change. This might look a bit odd in this case.
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class Classifier(nn.Module):
    def __init__(self, src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length, num_classes):
        super(Classifier, self).__init__()
        self.encoder = Encoder(src_vocab_size, embed_size, num_layers, heads, device, forward_expansion, dropout, max_length)
        self.classification_layer = nn.Linear(embed_size, num_classes)

    def forward(self, x, mask):
        out = self.encoder(x, mask)
        out = self.classification_layer(out[:, 0, :])  # use the output corresponding to the [CLS] token
        return out
