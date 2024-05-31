import subprocess
import ipfshttpclient
from flask import Flask, request
from flask_cors import CORS
from dataloader import TextTranDataset as DataLoader
from discrimloader import TextTranDataset as DiscrimLoader

app = Flask(__name__)
CORS(app)  # This is necessary to allow cross-origin requests from your React.js app

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data['message']

    # Here, you would typically call your language model to generate a response
    # For the sake of this example, we'll just echo back the user's message
    bot_message = 'You said: ' + user_message

    return {'message': bot_message}

@app.route('/upload_data', methods=['POST'])
def upload_data():
    data = request.get_json()

    # Connect to the local IPFS node
    client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')

    # Retrieve the datasets from IPFS
    dataloader_dataset = client.cat(data['dataloader_params']['ipfs_path'])
    discrimloader_dataset = client.cat(data['discrimloader_params']['ipfs_path'])

    # Create and configure the dataloader and discrimloader
    dataloader = DataLoader(dataloader_dataset)
    discrimloader = DiscrimLoader(discrimloader_dataset)

    # Call the training scripts for the generator and discriminator models
    subprocess.run(['python', 'textTrain.py', dataloader.dataset_path])
    subprocess.run(['python', 'textDiscrimTrain.py', discrimloader.dataset_path])

    return {'message': 'Data uploaded and training started'}

if __name__ == '__main__':
    app.run(debug=True)