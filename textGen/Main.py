import torch
import os
import json
import requests
from flask import Flask, flash, render_template
from web3 import Web3
from textGen import TextGenerator, TextTokenizer
from textGenDecoder import Decoder
from textDiscrim import TextTransformer
from textDiscrimEncoder import Encoder

# Load the configuration file
with open('config.json') as f:
    config = json.load(f)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY')
# Instantiate the transformer
transformer = TextTransformer()

# Instantiate the tokenizer
tokenizer = TextTokenizer()

# Instantiate the generator model
model = TextGenerator()

# Instantiate the discriminator model
textDiscrimEncoder = Encoder()

@app.route('/chat', methods=['POST'])
def chat():
    data = requests.get_json()
    user_message = data['message']

    # Use the transformer somewhere in your application
    transformed_text = transformer.transform(user_message)

    # Generate a response using the generator model
    generated_response = generate_text(tokenizer, model, transformed_text)

    # Classify the generated response using the discriminator model
    # Uncomment the following line if you have a discriminator model
    classification = classify_text(tokenizer, textDiscrimEncoder, generated_response)

    # If the discriminator classifies the response as real (or if you don't have a discriminator),
    # send the generated response; otherwise, send a default response
    # Uncomment the following lines if you have a discriminator model
    if classification == 1:
         bot_message = generated_response
    else:
         bot_message = 'I\'m sorry, I don\'t know how to respond to that.'

    # If you don't have a discriminator model, just send the generated response
    bot_message = generated_response

    return {'message': bot_message}

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input data from the request
    data = requests.get_json(force=True)

    # Pass the input to the textGenDecoder for inference
    try:
        output = Decoder.forward(data)
    except Exception as e:
        # If there's an error (e.g., the model isn't confident), flash the error message
        flash(str(e))
        output = None

    # Render a template with the prediction result
    return render_template('predict.html', prediction=output)

# Get the values from the configuration file
max_length = config['main']['max_length']
learning_rate = config['main']['learning_rate']
batch_size = config['main']['batch_size']

# Load the generator model
tokenizer = TextTokenizer()
model = TextGenerator()

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

def get_classification_from_another_source():
    # This is a placeholder URL; replace with the actual URL of the discriminator's web API
    url = 'http://localhost:5000/classification'
    response = requests.get(url)
    classification = response.json()['classification']
    return classification

# Define the main function
def main(filename, task):
    tokenizer = TextTokenizer(filename)

    if task == 'generate':
        # Load the generator model
        generator = TextGenerator()
        text = generate_text(tokenizer, generator)
        print(text)
    elif task == 'classify':
        # Get a classification from another source
        classification = get_classification_from_another_source()
        interact_with_blockchain(classification)
    else:
        print(f"Unknown task: {task}")

    text = generate_text()
    # Get a classification from another source
    classification = get_classification_from_another_source()
    interact_with_blockchain(classification)

if __name__ == "__main__":
    # Use a command line argument to specify the filename
    import sys
    if len(sys.argv) != 2:
        print("Usage: python main.py <filename>")
        sys.exit(1)
    main(sys.argv[1])