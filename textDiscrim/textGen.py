import json
import torch
import torch.nn as nn
import re
#import sys
import nltk
from torchtext.data import Field
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter

# Load the configuration from config.json
with open('config.json') as f:
    config = json.load(f)

class TextTokenizer(nn.Module):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.tokenizer = TextTokenizer()
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
        self.tokenizer.create_vocab_file(vocab_filename)
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
        super().__init__()
        self.embedding = nn.Embedding(config['vocab_size'], config['embedding_dim'])
        self.transformer = nn.Transformer(config['embedding_dim'], nhead=config['nhead'])
        self.fc = nn.Linear(config['embedding_dim'], config['num_classes'])
        self.tokenizer = tokenizer

    def forward(self, x, text):
        # Tokenize the text
        tokens = self.tokenizer.tokenize_text(text)

        x = self.embedding(tokens)
        x = self.transformer(x)
        x = x + self.positional_encoding(x)
        x = self.fc(x[-1])

        return x
    
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