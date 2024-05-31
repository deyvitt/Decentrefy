import ipfshttpclient
import torch
import io

class IPFSKon:
    def __init__(self):
        self.client = ipfshttpclient.connect('/ip4/127.0.0.1/tcp/5001/http')

    def download_weights(self, ipfs_hash):
        # Download the file from IPFS
        res = self.client.cat(ipfs_hash)

        # Convert the bytes to a dictionary
        weights = torch.load(io.BytesIO(res))

        return weights

    def upload_weights(self, weights):
        # Convert the weights to bytes
        weights_bytes = io.BytesIO()
        torch.save(weights, weights_bytes)
        weights_bytes.seek(0)

        # Upload the weights to IPFS
        res = self.client.add_pyobj(weights_bytes)

        return res
