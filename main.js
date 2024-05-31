import detectEthereumProvider from '@metamask/detect-provider';
//import { MutationObserver } from 'mutationobserver'; 

document.addEventListener('DOMContentLoaded', (_event) => {
  const connectButton = document.getElementById('connectButton');
  const submissionForm = document.getElementById('submissionForm');
  const dateDisplay = document.getElementById('dateDisplay');

  // Create a MutationObserver (replace with your desired logic)
  //const observer = new MutationObserver((mutations) => {
    //console.log('Mutations detected:', mutations);
    // You can access specific mutation details and handle changes here
    // For example:
    // mutations.forEach(mutation => {
    //   if (mutation.type === 'childList') {
    //     console.log('Child nodes added or removed!');
    //   }
    // });
  //});

  // Observe the document body for changes (replace with your target element)
  //observer.observe(document.body, { childList: true }); // Observe child element changes

  async function connectToWallet() {
    try {
      const provider = await detectEthereumProvider();
      if (provider) {
        console.log('Wallet detected!');
        const accounts = await provider.request({ method: 'eth_requestAccounts' });
        if (accounts.length > 0) {
          console.log('MetaMask connected with accounts:', accounts);
          window.web3 = new Web3(provider);
          submissionForm.style.display = 'block';
        } else {
          console.log('User rejected account access');
        }
      } else {
        console.log('No wallet detected.');
        alert('No compatible crypto wallet detected. Please install MetaMask!');
      }
    } catch (error) {
      console.error('Error connecting to wallet:', error);
    }
  }

  connectButton.addEventListener('click', connectToWallet);

  dateDisplay.innerText = new Date().toDateString();
  //document.getElementById('demo').innerHTML = new Date().toDateString();
});

