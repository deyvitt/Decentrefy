import json
import torch
import ipfsapi
import torch.nn as nn
from textDiscrim import TextTransformer

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
num_classes = config['num_classes']

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

# Create a Classifier model
model = Classifier(src_vocab_size, embed_size, num_layers, num_heads, device, forward_expansion, dropout, max_length, num_classes)

# Access the state dict of the model
state_dict = model.state_dict()

# The state dict is a dictionary mapping each layer to its weights and biases
for layer, params in state_dict.items():
    print(f"Layer: {layer}")
    print(f"Weights: {params}")

# Save the state dict to a file
torch.save(model.state_dict(), 'model_weights.pth')

# Connect to the local IPFS node
api = ipfsapi.connect('127.0.0.1', 5001)

# Add the file to IPFS
res = api.add('model_weights.pth')

# The IPFS hash of the file is in res['Hash']
print(f"Uploaded weights to IPFS with hash: {res['Hash']}")