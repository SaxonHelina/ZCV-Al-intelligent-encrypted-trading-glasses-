def get_project_financial_health(project_id):
    """Fetches the financial health of a project."""
    return contract.functions.getFinancialHealth(project_id).call()

def get_project_development_progress(project_id):
    """Fetches the development progress of a project."""
    return contract.functions.getDevelopmentProgress(project_id).call()

def get_project_community_feedback(project_id):
    """Fetches community feedback for a project."""
    return contract.functions.getCommunityFeedback(project_id).call()

import asyncio

def handle_event(event):
    """Handles events emitted by the smart contract."""
    print(f"New event: {event}")

async def log_loop(event_filter, poll_interval):
    """Continuously polls for new events."""
    while True:
        for event in event_filter.get_new_entries():
            handle_event(event)
        await asyncio.sleep(poll_interval)




# Smart contract ABI
contract_abi = [
    # ABI details here
]

# Smart contract address
contract_address = '0xYourContractAddressHere'

# Create contract instance
contract = web3.eth.contract(address=contract_address, abi=contract_abi)


def main():
    # Create an event filter for ProjectUpdated events
    event_filter = contract.events.ProjectUpdated.createFilter(fromBlock='latest')

    # Start the event loop
    loop = asyncio.get_event_loop()
    try:
        loop.run_until_complete(
            asyncio.gather(
                log_loop(event_filter, 2)
            )
        )
    except KeyboardInterrupt:
        print("Exiting...")
    finally:
        loop.close()

if __name__ == "__main__":
    main()
