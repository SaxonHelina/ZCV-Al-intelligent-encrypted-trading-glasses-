from web3 import Web3
import json

# Connect to Ethereum blockchain (replace with actual Ethereum node URL)
infura_url = 'https://mainnet.infura.io/v3/YOUR_INFURA_PROJECT_ID'
web3 = Web3(Web3.HTTPProvider(infura_url))

# Check if connected
print("Connected to Ethereum:", web3.isConnected())

# Load your wallet credentials (private key, address)
private_key = 'YOUR_PRIVATE_KEY'
account = web3.eth.account.privateKeyToAccount(private_key)

# Define the transaction contract ABI (simplified for illustration)
contract_abi = json.loads('[{"constant":true,"inputs":[],"name":"getTransactionCount","outputs":[{"name":"","type":"uint256"}],"payable":false,"stateMutability":"view","type":"function"}]')

# Define the contract address (example)
contract_address = 'YOUR_CONTRACT_ADDRESS'
contract = web3.eth.contract(address=contract_address, abi=contract_abi)

# Record a trade on the blockchain
def record_trade_on_blockchain(trade_details):
    nonce = web3.eth.getTransactionCount(account.address)
    transaction = contract.functions.getTransactionCount().buildTransaction({
        'from': account.address,
        'gas': 2000000,
        'gasPrice': web3.toWei('20', 'gwei'),
        'nonce': nonce,
    })

    # Sign the transaction
    signed_txn = web3.eth.account.signTransaction(transaction, private_key)

    # Send the transaction
    txn_hash = web3.eth.sendRawTransaction(signed_txn.rawTransaction)

    # Wait for transaction receipt
    receipt = web3.eth.waitForTransactionReceipt(txn_hash)
    print(f"Trade recorded on blockchain with transaction hash: {receipt.transactionHash.hex()}")

# Example of recording a trade
trade_details = {
    'action': 'BUY',
    'price': 150.0,
    'quantity': 10
}
record_trade_on_blockchain(trade_details)
