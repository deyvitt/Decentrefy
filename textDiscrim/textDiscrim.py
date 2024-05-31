import json
import torch
import torch.nn as nn
import re
import nltk
#import sys
from flask import Flask, flash, render_template
from torch.nn import MultiheadAttention
from torchtext.data import Field
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from itertools import chain
from textGenDecoder import Decoder
from textDiscrimEncoder import Encoder
from confidenceScore import ConfidenceScorer

# Load the configuration from config.json
with open('config.json') as f:
    config = json.load(f)

# Get the values from the configuration file
filename = config['filename']
vocab_filename = config['vocab_filename']
learning_rate = config['learning_rate']
batch_size = config['batch_size']

class TextDiscrim:
    def __init__(self, model):
        self.model = model
        self.confidence_scorer = ConfidenceScorer(self.model)
        self.load_config()

    def get_confidence_score(self, user_prompt):
        return self.confidence_scorer.score(user_prompt)

class TextTokenizer(nn.Module):
    def __init__(self, filename):
        super().__init__()
        filename = os.path.basename(filename)
        self.filename = os.path.join('safe-directory', filename)
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        # Define a Field that tokenizes text using the 'basic_english' tokenizer
        self.TEXT = Field(tokenize=word_tokenize, lower=True, batch_first=True, preprocessing=self.custom_preprocessing)
        self.vocab = Counter()

    def process_texts(self):
        with open(self.filename, 'r') as f:
            texts = f.read().split('\n')

        for text in texts:
            self.tokenizer.tokenize_text(text)

    def create_vocab_file(self, vocab_filename):
#       self.tokenizer.create_vocab_file(vocab_filename)
        vocab_dict = dict(self.vocab)
        with open(vocab_filename, 'w') as f:
            json.dump(vocab_dict, f)

    def custom_preprocessing(self, text):
        # Standardize '&' with 'and'
        text = text.replace('&', ' and ')
        text = text.replace('@', ' at ')
        text = re.sub(r'http\S+|www.\S+', 'url', text)
        text = re.sub(r'\S+@\S+', 'email', text)

        # Convert to lowercase
        text = text.lower()
        
        # Tokenize the text into words
        words = nltk.word_tokenize(text)
        
        # Remove punctuation
        words = [re.sub(r'[^\w\s]', '', word) for word in words]
        
        # Remove stopwords
        words = [word for word in words if word not in self.stop_words]
        
        # Lemmatize words
        words = [self.lemmatizer.lemmatize(word) for word in words]

        # Update the vocabulary with the words from this text
        self.vocab.update(words)
        
        return words

    def tokenize_text(self, text):
        # Use the Field to process the text
        tokens = self.TEXT.process([self.TEXT.preprocess(text)])
        return tokens
        
class TextTransformer(nn.Module):
    def __init__(self, config, tokenizer):
        super(TextTransformer, self).__init__()
        self.sliding_window_attention = MultiheadAttention(config['embedding_dim'], config['nhead'], dropout=config['dropout'])
        self.global_attention = MultiheadAttention(config['embedding_dim'], config['nhead'], dropout=config['dropout'])
        self.feed_forward = nn.Sequential(
            nn.Linear(config['embedding_dim'], config['forward_expansion'] * config['embedding_dim']),
            nn.ReLU(),
            nn.Linear(config['forward_expansion'] * config['embedding_dim'], config['embedding_dim']),
        )
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.encoder = Encoder(config)
        self.fc = nn.Linear(config['embedding_dim'], config['num_classes'])
        self.tokenizer = tokenizer

    def forward(self, x, text, query, key, value, mask):
        sliding_window_output, _ = self.sliding_window_attention(query, key, value, mask)
        global_output, _ = self.global_attention(query, key, value, mask)
        # combine the outputs in some way, for example by adding them
        output = sliding_window_output + global_output

        # Tokenize the text
        tokens = self.tokenizer.tokenize_text(text)

        x = self.embedding(tokens)
        x = self.encoder(x)
#       x = self.decoder(x)
#       x = self.transformer(x)
        x = x + self.positional_encoding(x)
        x = self.fc(x[-1])

        return x, output
    
    def positional_encoding(self, x):
        sequence_length = x.size(1)
        embedding_size = x.size(2)
        pos = torch.arange(sequence_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_size, 2).float() * -(torch.log(torch.tensor(10000.0)) / embedding_size))
        pos_enc = torch.zeros(sequence_length, embedding_size)
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        return x + pos_enc.to(x.device)    

if __name__ == "__main__":
    # This code will only be run when textTransformer.py is executed as a standalone program
    tokenizer = TextTokenizer('data.txt')
    tokenizer.process_texts()
    tokenizer.create_vocab_file('vocab.json')
    transformer = TextTransformer(config, tokenizer)

    app.run(debug=True)

# Define your training data
training_data = ["This is the first sentence.", "This is another sentence.", "And this is the last one."]  # replace with your actual data

# Define the maximum number of words in your vocabulary
VOCAB_SIZE = 10000  # replace with your actual vocabulary size

# Get a list of all words in your training data
words = list(chain(*[word_tokenize(text) for text in training_data]))

# Count the frequency of each word
word_counts = Counter(words)

# Create a vocabulary by sorting the words by frequency and taking the top N words
vocabulary = [word for word, _ in word_counts.most_common(VOCAB_SIZE)]

# Create a dictionary that maps each word in the vocabulary to a unique integer
word_to_int = {word: i for i, word in enumerate(vocabulary)}

# Now you can use word_to_int to convert words to integers
tokens = [word_to_int[word] for text in training_data for word in word_tokenize(text) if word in word_to_int]