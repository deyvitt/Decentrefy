// Solidity
pragma solidity ^0.8.0;

contract SubscriptionContract {
    function payMonthlyFee() public payable {
        require(msg.value == 0.003 ether, "You must pay exactly 0.003 ETH");
        // Store the payment and the time of payment for the user
    }
} 