import torch
import os
#import ipfsapi
import torch.nn as nn
import json
from textGen import TextTransformer
from confidenceScore import ConfidenceScorer

# Load the configuration from config.json
with open('config.json') as f:
    config = json.load(f)

# Get the values from the configuration file
tgt_vocab_size = config['textGenDecoder']['tgt_vocab_size']
embed_size = config['textGenDecoder']['embed_size']
num_layers = config['textGenDecoder']['num_layers']
num_heads = config['textGenDecoder']['num_heads']
device = config['textGenDecoder']['device']
forward_expansion = config['textGenDecoder']['forward_expansion']
dropout = config['textGenDecoder']['dropout']
max_length = config['textGenDecoder']['max_length']

class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)
        self.device = device
        self.transformer_model = TextTransformer(
            embed_size,
            heads,
            dropout=dropout,
            forward_expansion=forward_expansion,
        )
        
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.hybrid_attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_model(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embed_size, num_layers, heads, forward_expansion, dropout, device,max_length):
        self.decoder = Decoder(
            trg_vocab_size=config['trg_vocab_size'],
            embed_size=config['embed_size'],
            num_layers=config['num_layers'],
            num_heads=config['heads'],
            forward_expansion=config['forward_expansion'],
            dropout=config['dropout'],
            device=config['device'],
            max_length=config['max_length'],
            device=device,
        )
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.fc_out = nn.Linear(embed_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask, trg, trg_mask):
        # Apply hybrid attention
        self.decoder_output, _ = self.hybrid_attention(trg, encoder_output, encoder_output, attn_mask=trg_mask)
    
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        # Generate a square subsequent mask for the source sequence
        src_mask = self._generate_square_subsequent_mask(seq_length).to(self.device)

        for layer in self.layers:
            x = layer(x, encoder_output, encoder_output, src_mask, trg_mask)

        out = self.fc_out(x)

        return out

    def Activate_TextDiscrim(self, x, encoder_output, src_mask, trg, trg_mask, textDiscrim, textDiscrimEncoder):
        # Generate output from textGenDecoder
        output = self.forward(x, encoder_output, src_mask, trg, trg_mask)

        # Calculate the confidence score
        confidence_score = ConfidenceScorer(output)

        # Only pass the output to textDiscrim if the confidence score is high enough
        if confidence_score == 1:  
            textDiscrim_output = textDiscrim(output)

            # Pass the output to textDiscrimEncoder
            textDiscrimEncoder_output = textDiscrimEncoder(textDiscrim_output)
        else:
            textDiscrimEncoder_output = None 
            print("This model couldn't respond to your prompt, can you re-phrase?")  

        return textDiscrimEncoder_output
    
"""
Since we are passing the textGenDecoder output to textDiscrimEncoder, these snippets aren't needed:
____________________________________________________________
# Create an instance of TextTransformer
textGen = TextTransformer(config[embed_size], config[num_heads], config[dropout], config[forward_expansion])

# Access the state dict of the generator model
state_dict = TextTransformer.state_dict()

# The state dict is a dictionary mapping each layer to its weights and biases
for layer, params in state_dict.items():
    print(f"Layer: {layer}")
    print(f"Weights: {params}")

# Save the state dict to a file
torch.save(TextTransformer.state_dict(), 'gen_model_weights.pth')

# Connect to the local IPFS node
api = ipfsapi.connect('127.0.0.1', 5001)

# Add the file to IPFS
res = api.add('gen_model_weights.pth')

# The IPFS hash of the file is in res['Hash']
print(f"Uploaded generator weights to IPFS with hash: {res['Hash']}")
"""