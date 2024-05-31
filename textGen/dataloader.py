from torchtext.data import Field, TabularDataset, BucketIterator
from torch.optim import Adam
from textTrain import EarlyStopping
from textGen import TextTransformer
import torch.nn as nn
import torch
import json

# Define your configuration parameters here
# Load the configuration from config.json
with open('config.json') as f:
    config = json.load(f)

# Get the values from the configuration file
max_size = config['dataloader']['max_size']
batch_size = config['dataloader']['batch_size']
csv_path = config['dataloader']['csv_path']
patience = config['dataloader']['patience']

class TextTranDataset:
    def __init__(self, num_epochs):
        # Define the fields
        self.TEXT = Field(sequential=True, tokenize='spacy', lower=True)
        self.LABEL = Field(sequential=False, use_vocab=False)
        self.early_stopping = EarlyStopping(patience=7, verbose=True, monitor='val_loss', mode='min')

        # Create the dataset
        self.dataset = TabularDataset(path='path_to_your_file.csv', format='csv', fields=[('text', self.TEXT), ('label', self.LABEL)])

        # Build the vocabulary
        self.TEXT.build_vocab(self.dataset, max_size=config['max_size'])
        self.LABEL.build_vocab(self.dataset)

        # Create the data loader
        dataloader = BucketIterator(dataloader.dataset, batch_size=32, sort_key=lambda x: len(x.text), sort_within_batch=True)

        # Define the model
        self.model = TextTransformer(input_dim=len(self.TEXT.vocab), output_dim=len(self.LABEL.vocab))

        # Define the loss function and optimizer
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.model.parameters())

        # Move model and loss function to GPU if available
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.loss_function = self.loss_function.to(self.device)
        # Initiate num_epochs
        self.num_epochs = num_epochs

