import React, { useState } from 'react';
import { create } from 'ipfs-http-client';
import web3 from 'web3';
import { Box, Button, Input, Text, Flex } from "@chakra-ui/react";
import 'react-dropzone-uploader/dist/styles.css';
import Dropzone from 'react-dropzone-uploader';

import decentrefy from './Decentrefy.json';

const ipfs = create({ host: 'ipfs.infura.io', port: 5001, protocol: 'https' });
const decentrefyAddress = " ";

const MainMint = ({ accounts, setAccounts }) => {
    const [mintAmount, setMintAmount] = useState(1);
    const [uploadedData, setUploadedData] = useState(null);
    const [setFile] = useState(null);   
    const [setFileData] = useState(null); 
    const isConnected = Boolean(accounts[0]);

    async function handleFileChange(event) {
        const file = event.target.files[0];
        setFile(file);  

        // Convert the file to a blob
        const fileData = await file.arrayBuffer();
        setFileData(fileData);
        const blob = new Blob([fileData]);
    
        // Add the file to IPFS
        const result = await ipfs.add(blob);
        console.log('IPFS path: ', result.path);
        setUploadedData(result.path);

        // Create a FormData instance
        const formData = new FormData();
        // Append the file to the form data
        formData.append('file', file);

        // Send the file to the Python server
        fetch('http://localhost:5000/upload_data', {
            method: 'POST',
            body: formData,
        })
        .then(response => response.json())
        .then(data => console.log(data))
        .catch((error) => console.error('Error:', error));
    }
    
    function generateMetadata(mintAmount) {
        return {
            name: `Token ${mintAmount}`,
            description: `Description for token ${mintAmount}`,
            image: `ipfs://image/path/for/token/${mintAmount}`,
            unlockable: `As the owner of the token ${mintAmount}, you store your datasets in 'unlockable' feature. Visit https://decentrefy.com/bonus`
        };
    }

    const handleDrop = async ({ meta }, status) => {
        if (status === 'done') {
            // meta contains file metadata
            const fileData = await fetch(meta.previewUrl).then(r => r.blob());
            const result = await ipfs.add(fileData);
            console.log('IPFS path: ', result.path);
            setUploadedData(result.path);

            // Send result.path to Python dataloader modules
            const data = {
                dataloader_params: { ipfs_path: result.path },
                discrimloader_params: { ipfs_path2: result.path },  // Replace with actual parameters
            };

            fetch('http://localhost:5000/upload_data', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data),
            })
            .then(response => response.json())
            .then(data => console.log(data))
            .catch((error) => console.error('Error:', error));    
        }
    };
    
    async function handleMint() {
        // Generate the metadata for the token
        const metadata = generateMetadata(mintAmount);

        // Include the IPFS path to the training data in the token's metadata
        metadata.trainingDataPath = uploadedData;

        // Add the metadata to IPFS
        const { path } = await ipfs.add(JSON.stringify(metadata));

        const web3Instance = new web3(window.ethereum);
        const contract = new web3Instance.eth.Contract(
            decentrefy.abi,
            decentrefyAddress
        );
        try {
            const response = await contract.methods.mint(web3.utils.toBN(mintAmount), path).send({
                from: accounts[0],
                value: web3.utils.toWei((0.02 * mintAmount).toString(), 'ether'),
            });
            console.log('response: ', response);
            // Update accounts state after successful mint
            setAccounts(accounts);
        } catch (err) {
            console.log("error: ", err)
        }
    }

    const handleDecrement = () => {
        if (mintAmount <= 1) return;
        setMintAmount(mintAmount - 1);
    };

    const handleIncrement = () => {
        if (mintAmount >= 3) return;
        setMintAmount(mintAmount + 1);
    };

    return (
        <Flex justify="center" align="center" height="100vh" paddingBottom="150px">
            <Box width="520px">
                <div>
                    <Text fontSize="48px" textShadow=' 5px #000000'>Decentrefy</Text>
                    <Text
                        fontSize="30px"
                        letterSpacing="-5.5%"
                        fontFamily="VT323"
                        textShadow="0 2px 2px #000000"
                    >
                        Help us train our language model and get rewarded</Text>
                </div>

                {isConnected ? (
                    <div>
                        <Flex align="center" justify="center">
                            <Button 
                                backgroundColor="#D6517D"
                                borderRadius="5px"
                                boxShadow="0px 2px 2px 1px #0F0F0F"
                                color="white"
                                cursor="pointer"
                                padding="15px"
                                marginTop="10px"
                                onClick={handleDecrement}
                            >
                                -
                            </Button>
                            <Input 
                                readOnly
                                fontFamily="inherit"
                                width="100px"
                                height="40px"
                                textAlign="center"
                                paddingLeft="19px"
                                marginTop="10px"
                                type="number"
                                value={mintAmount}
                            />
                            <Button 
                                backgroundColor="#D6517D"
                                borderRadius="5px"
                                boxShadow="0px 2px 2px 1px #0F0F0F"
                                color="white"
                                cursor="pointer"
                                padding="15px"
                                marginTop="10px"
                                onClick={handleIncrement}
                            >
                                +                       
                            </Button>
                        </Flex>
                        <Button 
                            backgroundColor="#D6517D"
                            borderRadius="5px"
                            boxShadow="0px 2px 2px 1px #0F0F0F"
                            color="white"
                            cursor="pointer"
                            padding="15px"
                            marginTop="10px"
                            onClick={handleMint}
                        >
                            Mint Now
                        </Button>
                        <Dropzone
                            inputContent="Drop Files Here"
                            onChangeStatus={handleDrop}
                        />
                        <input type="file" onChange={handleFileChange} />
                    </div>
                ) : (
                    <Text>You must be connected to Mint.</Text>
                )}
            </Box>
        </Flex>
    );
};

export default MainMint; 