import re


def tokenize(log):
    """
    Tokenize the log message by whitespace.
    """
    return log.strip().split()


def calc_similarity(tokens, template):
    """
    Calculate similarity between tokens and a template.
    For positions where the template has a wildcard "*", it is considered a match.
    Similarity = number of matches / total number of tokens.
    Returns 0 if token lengths do not match.
    """
    if len(tokens) != len(template):
        return 0.0
    match = 0
    for t, tmpl in zip(tokens, template):
        if tmpl == "*" or t == tmpl:
            match += 1
    return match / len(tokens)


class LogCluster:
    """
    Represents a log cluster, i.e., a log template.
    """

    def __init__(self, template_tokens, cluster_id):
        self.template_tokens = template_tokens[:]  # Initialize template with token list
        self.cluster_id = cluster_id
        self.size = 1  # Number of logs in the cluster
        self.log_ids = []  # Optionally store log indices or original logs

    def update(self, tokens):
        """
        Update the template based on new tokens.
        If a token doesn't match, replace with a wildcard "*".
        """
        new_template = []
        for i in range(len(self.template_tokens)):
            if self.template_tokens[i] == tokens[i]:
                new_template.append(self.template_tokens[i])
            else:
                new_template.append("*")
        self.template_tokens = new_template
        self.size += 1

    def __str__(self):
        return f"Cluster {self.cluster_id}: template: {' '.join(self.template_tokens)}, count: {self.size}"


class Node:
    """
    Tree node for the Drain algorithm.
    Each node holds its children and a list of log clusters under that node.
    """

    def __init__(self):
        self.children = {}  # key: token, value: Node instance
        self.clusters = []  # List of LogCluster objects


class DrainParser:
    """
    Main parser for the Drain algorithm.

    Parameters:
      - max_depth: maximum depth of the tree.
      - sim_th: similarity threshold for matching.
      - regex_list: list of tuples (pattern, replacement) for regex-based token replacement.
    """

    def __init__(self, max_depth=4, sim_th=0.4, regex_list=None):
        self.max_depth = max_depth
        self.sim_th = sim_th
        self.root = Node()
        self.clusters = []  # Global list of clusters
        self.cluster_count = 0  # Counter for cluster IDs
        self.regex_list = regex_list if regex_list else []

    def replace_regex(self, token):
        """
        Apply all regex replacements to a token.
        """
        for pattern, replacement in self.regex_list:
            token = re.sub(pattern, replacement, token)
        return token

    def add_log_message(self, log):
        """
        Process a log message:
          1. Tokenize and apply regex replacements.
          2. Traverse the tree using the first max_depth tokens.
          3. Find the most similar cluster in the node.
          4. If similarity is above threshold, update the cluster; otherwise, create a new cluster.
        """
        tokens = tokenize(log)
        tokens = [self.replace_regex(token) for token in tokens]

        depth = min(self.max_depth, len(tokens))
        node = self.root
        for i in range(depth):
            token = tokens[i]
            if token not in node.children:
                node.children[token] = Node()
            node = node.children[token]

        matched_cluster = None
        max_sim = -1.0
        for cluster in node.clusters:
            sim = calc_similarity(tokens, cluster.template_tokens)
            if sim > max_sim:
                max_sim = sim
                matched_cluster = cluster

        if max_sim >= self.sim_th:
            matched_cluster.update(tokens)
        else:
            new_cluster = LogCluster(tokens, self.cluster_count)
            self.cluster_count += 1
            node.clusters.append(new_cluster)
            self.clusters.append(new_cluster)

    def parse_logs(self, logs):
        """
        Parse a list of log messages.
        """
        for log in logs:
            self.add_log_message(log)

    def get_templates(self):
        """
        Return the templates of all clusters.
        """
        return [cluster.template_tokens for cluster in self.clusters]


# ---------------------------
# Test Example
# ---------------------------
if __name__ == '__main__':
    logs = [
        "Error: Disk /dev/sda1 is full",
        "Error: Disk /dev/sda2 is full",
        "Warning: CPU temperature too high",
        "Error: Disk /dev/sda1 is full",
        "Warning: CPU temperature at 95",
        "Info: User John logged in",
        "Info: User Mike logged in",
        "Info: User John logged out",
        "Error: Disk /dev/sda3 is full"
    ]
    # Define regex rules to replace disk names and numbers with placeholders
    regex_list = [
        (r'/dev/sda\d+', '<disk>'),
        (r'\d+', '<num>')
    ]
    parser = DrainParser(max_depth=3, sim_th=0.5, regex_list=regex_list)
    parser.parse_logs(logs)

    print("Extracted log templates:")
    for cluster in parser.clusters:
        print(cluster)


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
