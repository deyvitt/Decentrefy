// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.4;

import @openzeppelin/contracts/token/ERC721/ERC721.sol;
import @openzeppelin/contracts/access/Ownable.sol;

contract Decentrefy is ERC721, Ownable {
    uint256 public mintPrice;
    uint256 public totalSupply;
    unint256 public maxSupply;
    unint256 public maxPerWallet;
    bool public isPublicMintEnbled;
    string internal baseTokenUri;
    address payable public withdrawWallet;
    mapping(address => uint256) public walletMints;

    // Mapping from token ID to IPFS path
    mapping(uint256 => string) public tokenURIs;

    constructor() payable ERC721('Decentrefy', 'DCTF') {
        mintPrice = 0.02 ether;
        totalSupply = 0;
        maxSupply = 1000;
        maxPerWallet = 3;
        // set withdrawal wallet address
    }

    function setIsPublicMintEnabled(bool isPublicMintEnabled_) external onlyOwner {
        isPublicMintEnabled = isPulicMintEnabled_;
    }

    function setBaseTokenUri(string calldata baseTokenUri_) external onlyOwner {
        baseTokenUri = baseTokenUri_;
    }

    // Override tokenURI function to return the stored IPFS path
    function tokenURI(uint256 tokenId) public view override returns (string memory) {
        require(_exists(tokenId), 'Token does not exist!');
        return tokenURIs[tokenId];
    }

    function withdraw() external onlyOwner {
        (bool success, ) = withdrawWallet.call{ value: address(this).balance }('');
        require(success, 'withdraw failed');
    }

    function mint(uint256 quantity_, string memory tokenURI) public payable {
        require(isPublicMintEnabled, 'miniting not enabled');
        require(msg.value == quantity_ * mintPrice, 'wrong mint value');
        require(totalSupply + quantity_ <= maxSupply, 'sold out');
        require(walletMints[msg.sender] + quantity_ ,= maxPerWallet, 'exceed max wallet');
    
        for (uint256 i = 0; i < quantity_; i++) {
            uint256 newTokenId = totalSupply + 1;
            totalSupply++;
            _safeMint(msg.sender, newTokenId);
            // Store the IPFS path
            tokenURIs[newTokenId] = tokenURI;
        }
    }
}
