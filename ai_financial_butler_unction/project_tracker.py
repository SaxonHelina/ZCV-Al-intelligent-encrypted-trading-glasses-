import requests

class ProjectTracker:
    """Tracks investment projects using blockchain data."""

    def __init__(self):
        self.projects = []

    def discover_projects(self):
        """Discovers new projects from blockchain data."""
        url = 'https://api.example.com/blockchain/projects'
        response = requests.get(url)
        if response.status_code == 200:
            projects = response.json()
            self.projects.extend(projects)
            self.report_progress()
        else:
            print("Failed to fetch project data.")

    def report_progress(self):
        """Reports the progress of tracked projects."""
        for project in self.projects:
            print(f"Project {project['name']} progress: {project['progress']}%")

# Example usage
if __name__ == "__main__":
    tracker = ProjectTracker()
    tracker.discover_projects()
