const hre = require("hardhat");

async function main() {
    const Decentrefy = await hre.web3.getContractFactory("Decentrefy");
    const decentrefy = await Decentrefy.deploy();

    await decentrefy.deployed();

    console.log("Decentrefy deployed to:", decentrefy.address);
}

// it's recommended that this pattern to be able to use async/await everywhere
// and to properly handle errors.
main()
    .then(() => process.exit(0))
    .catch((error) => {
        console.error(error);
        process.exit(1);
    });