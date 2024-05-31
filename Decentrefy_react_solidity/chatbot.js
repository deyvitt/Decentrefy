import React, { useState, useEffect } from 'react';
import Web3 from 'web3';
//import { useWeb3React } from '@web3-react/core';
//import { InjectedConnector } from '@web3-react/injected-connector';
import './Chatbot.css';

// Create a new injected connector
//const injected = new InjectedConnector({ supportedChainIds: [1, 3, 4, 5, 42] });

// ABI of your contract (replace with your actual ABI)
const contractABI = [];

// Address of your contract (replace with your actual contract address after deploying on Ethereum)
const contractAddress = '0x...';

function Chatbot() {
    const [message, setMessage] = useState('');
    const [chatHistory, setChatHistory] = useState([]);
   // const { activate, active } = useWeb3React();
    const [web3, setWeb3] = useState(null);
    const [account, setAccount] = useState('');

    useEffect(() => {
        if (window.ethereum) {
            const web3Instance = new Web3(window.ethereum);
            setWeb3(web3Instance);
        } else {
            alert('Please install MetaMask!');
        }
    }, []);

    const handleChargeUser = async () => {
        if (web3 && account) {
            // Create a new contract instance
            const contract = new web3.eth.Contract(contractABI, contractAddress);

            // Call the chargeUser function of the contract
            const chargeAmount = web3.utils.toWei('0.003', 'ether');
            await contract.methods.chargeUser(account, chargeAmount).send({ from: account });
        }
    };
    const handleSendMessage = () => {
        // Here you would normally send the message to your chatbot backend
        // and get the response. For this example, we'll just echo the user's message.
        setChatHistory([...chatHistory, { sender: 'You', message }, { sender: 'Bot', message }]);
        setMessage('');
    };

    const handleConnectWallet = async () => {
        if (web3) {
            const accounts = await window.ethereum.request({ method: 'eth_requestAccounts' });
            setAccount(accounts[0]);
        }
    };

    return (
        <div>
            <button onClick={handleConnectWallet} className="connect-button">
                {account ? `Connected: ${account}` : 'Connect Wallet'}
            </button>
            <button onClick={handleChargeUser}>Charge User</button>
            <div className="chat-history">
                {chatHistory.map((chat, index) => (
                    <p key={index}><strong>{chat.sender}:</strong> {chat.message}</p>
                ))}
            </div>
            <input
                type="text"
                value={message}
                onChange={e => setMessage(e.target.value)}
                onKeyDown={e => e.key === 'Enter' ? handleSendMessage() : null}
                className="chat-input"
            />
            <button onClick={handleSendMessage} className="send-button">Send</button>
        </div>
    );
}

export default Chatbot; 