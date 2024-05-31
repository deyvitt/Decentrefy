import torch
import json
import os
from web3 import Web3
from textGen import TextGenerator
from textDiscrim import TextTransformer, TextTokenizer
#whafrom nltk.tokenize import word_tokenize

# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

# Get the MAX_LENGTH value from the configuration file
max_length = config['max_length']
learning_rate = config['learning_rate']
batch_size = config['batch_size']

# Load the generator and discriminator models
generator = TextGenerator()
discriminator = TextTransformer()

# Connect to the Ethereum/Polygon blockchain
w3 = Web3(Web3.HTTPProvider("http://localhost:8545"))  # replace with your provider

# Define a function for generating text
def generate_text(tokenizer, generator, prompt):
    # Convert the prompt to tokens
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    # Generate text using the generator model
    with torch.no_grad():
        output = generator.generate(input_ids, max_length=max_length)

    # Decode the output tokens to text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Define a function for classifying text
def classify_text(tokenizer, discriminator, text):
    # Convert the text to tokens
    input_ids = torch.tensor([tokenizer.encode(text)])

    # Classify the text using the discriminator model
    with torch.no_grad():
        logits = discriminator(input_ids)

    # Convert the logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Get the predicted class
    predicted_class = torch.argmax(probabilities).item()
    return predicted_class

# Connect to the Ethereum network
w3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER')))  # get provider from environment variable

# Define a function for interacting with the blockchain
def interact_with_blockchain(classification):
    # Get the recipient address and the private key from environment variables
    to_address = os.getenv('TO_ADDRESS')
    private_key = os.getenv('PRIVATE_KEY')

    # Set the amount to send based on the classification
    if classification == 0:
        amount = w3.toWei(1, 'ether')
    else:
        amount = w3.toWei(0.1, 'ether')

    # Get the nonce
    nonce = w3.eth.getTransactionCount(w3.eth.accounts[0])

    # Prepare the transaction
    txn = {
        'nonce': nonce,
        'to': to_address,
        'value': amount,
        'gas': 2000000,
        'gasPrice': w3.toWei('50', 'gwei')
    }

    # Sign the transaction
    signed_txn = w3.eth.account.signTransaction(txn, private_key)

    # Send the transaction
    tx_hash = w3.eth.sendRawTransaction(signed_txn.rawTransaction)

    # Get the transaction receipt
    tx_receipt = w3.eth.waitForTransactionReceipt(tx_hash)

    return tx_receipt

# Define the main function
def main(filename, task):
    # Initialize the tokenizer
    tokenizer = TextTokenizer(filename)

    if task == 'classify':
        # Load the discriminator model
        discriminator = TextTransformer()
        with open('generated_text.txt', 'r') as f:
            text = f.read()
        classification = classify_text(tokenizer, discriminator, text)
        interact_with_blockchain(classification)
    else:
        print(f"Unknown task: {task}")

    interact_with_blockchain(classification)

if __name__ == "__main__":
    # Use a command line argument to specify the filename
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <filename>")
        sys.exit(1)
    main(sys.argv[1])