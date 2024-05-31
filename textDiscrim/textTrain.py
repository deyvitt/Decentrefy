# Python
import json
import torch
import torch.nn as nn
from torch.optim import Adam
from torchtext.data import Field, TabularDataset, BucketIterator
from textTransformer import TextTransformer

# Load the configuration from config.json
with open('config.json') as f:
    config = json.load(f)

class EarlyStopping:
    # EarlyStopping class to monitor validation metric and stop training 
    # when it stops improving for a certain patience window.
    def __init__(self):
        with open('C:/SwarmLLM/Orchestrator/config.json') as f:
            self.config = json.load(f)
        self.patience=self.config['early_stopping_patience'],
        self.verbose=self.config['early_stopping_verbose'],
        self.monitor=self.config['early_stopping_monitor'],
        self.mode=self.config['early_stopping_mode']
        self.best_score = None
        self.wait = 0
        self.stopped_epoch = 0
        self.early_stop = False

    def __call__(self, epoch, logs):
        current_score = logs.get(self.monitor)
        if current_score is None:
            # If the monitored metric is not available, skip early stopping check.
            return False

        if self.best_score is None or \
            (self.mode == 'min' and current_score <= self.best_score) or \
            (self.mode == 'max' and current_score >= self.best_score):
            self.best_score = current_score
            self.wait = 0
        else:
            # If current score is not better (depending on mode), increment wait counter.
            self.wait += 1
            if self.wait >= self.patience:
                # If current score is better, update best score and reset wait counter.
                self.stopped_epoch = epoch
                self.early_stop = True
                if self.verbose:
                    print(f'EarlyStopping: Stop training at epoch {self.stopped_epoch}')

                # If wait counter reaches patience, stop training.
                return True
            else:
                return False

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

class TextTransTrainer:
    def train(self, data_loader, epoch):
        self.model.train()
        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(data_loader)
        val_loss = self.validate(data_loader,epoch)
        if self.early_stopping(epoch, {'val_loss': val_loss}):
            print('Early stopping triggered')
            return avg_loss, True

        return avg_loss, False

    def validate(self, data_loader, epoch):
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(data_loader, 0):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                avg_loss = running_loss / len(data_loader)

        val_loss = running_loss / len(self.val_loader)
        print(f'Val Epoch: {epoch}, Loss: {val_loss}')
        return val_loss, avg_loss


    def run(self, train_loader, val_loader, epochs, checkpoint_path):
        best_val_loss = float('inf')
        for epoch in range(epochs):
            train_loss = self.train(train_loader)
            val_loss = self.validate(val_loader)
            print(f'Epoch {epoch+1}/{epochs} Train Loss: {train_loss} Val Loss: {val_loss}')
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(epoch, train_loss, checkpoint_path.format(epoch))
        print('Finished Training')

    def save_model(self, epoch, loss, checkpoint_path):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, checkpoint_path)

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    def freeze_layers(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_last_layer(self):
        for param in self.model.fc.parameters():
            param.requires_grad = True

def main():
    # Defining these variables
    db_name = 'your_database_name'
    user = 'your_username'
    password = 'your_password'
    training_data = []  # Initialize as an empty list, or load from somewhere

    model = TextTransformer()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Create an instance of TextLoader
    loader = TextTranDataset(db_name, user, password)

    # Load the 'truly unique' words
    knowledge_graph_data = loader.load_knowledge_graph_data()

    # Add the 'truly unique' words to your training data
    training_data.extend(knowledge_graph_data)
    trainer = TextTransTrainer(model, criterion, optimizer)

    train_data, val_data = TextTransTrainer.load_data()
    train_loader = TextTranDataset(train_data, batch_size=32, shuffle=True)
    val_loader = TextTranDataset(val_data, batch_size=32, shuffle=False)

    epochs = 20 # Need to see if we need to increase the number of epochs by monitoring model performance  
    last_loss = 1.0 # Initialize higher than threshold

    while last_loss > 0.01:
        trainer.run(train_loader, val_loader, epochs, 'checkpoint_{}.pth')
        trainer.freeze_layers()
        trainer.unfreeze_last_layer()
        last_loss = trainer.run(train_loader, val_loader, epochs, 'fine_tuned_checkpoint_{}.pth')

        if last_loss > 0.01:
            print(f'Loss is {last_loss}, need to continue training')
        else:
            print('Loss is low, stopping training')
    trainer.save_model(epochs, last_loss, 'final_model.pth')
    return last_loss

if __name__ == "__main__":
    final_loss = main()
    print(f'Final loss: (final_loss)')

# Create the TextTranDataset
text_tran_dataset = TextTranDataset(num_epochs=config['num_epochs'])

# Create the data loader
dataloader = BucketIterator(text_tran_dataset.dataset, batch_size=32, sort_key=lambda x: len(x.text), sort_within_batch=True)
