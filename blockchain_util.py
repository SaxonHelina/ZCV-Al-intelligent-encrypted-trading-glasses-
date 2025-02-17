import hashlib
import time
import json

# Block Class
class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash

# Blockchain Class
class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    # Create the first block (genesis block)
    def create_genesis_block(self):
        genesis_block = Block(0, "0", int(time.time()), "Genesis Block", self.calculate_hash(0, "0", int(time.time()), "Genesis Block"))
        self.chain.append(genesis_block)

    # Add a block to the blockchain
    def add_block(self, data):
        previous_block = self.chain[-1]
        new_block = Block(len(self.chain), previous_block.hash, int(time.time()), data, self.calculate_hash(len(self.chain), previous_block.hash, int(time.time()), data))
        self.chain.append(new_block)

    # Calculate hash for a block
    def calculate_hash(self, index, previous_hash, timestamp, data):
        block_string = str(index) + previous_hash + str(timestamp) + data
        return hashlib.sha256(block_string.encode('utf-8')).hexdigest()

    # Verify the integrity of the blockchain
    def verify_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            # Check if the hash of the current block matches the calculated hash
            if current_block.hash != self.calculate_hash(current_block.index, current_block.previous_hash, current_block.timestamp, current_block.data):
                return False

            # Check if the previous hash matches the previous block's hash
            if current_block.previous_hash != previous_block.hash:
                return False

        return True

    # Print the blockchain
    def print_blockchain(self):
        for block in self.chain:
            print(f"Block {block.index}:")
            print(f"Previous Hash: {block.previous_hash}")
            print(f"Timestamp: {block.timestamp}")
            print(f"Data: {block.data}")
            print(f"Hash: {block.hash}\n")

# Example usage:
if __name__ == "__main__":
    # Create a new blockchain
    my_blockchain = Blockchain()

    # Add new blocks to the blockchain
    my_blockchain.add_block("First block data")
    my_blockchain.add_block("Second block data")
    my_blockchain.add_block("Third block data")

    # Print the blockchain
    my_blockchain.print_blockchain()

    # Verify the blockchain's integrity
    print("Is blockchain valid?", my_blockchain.verify_chain())
