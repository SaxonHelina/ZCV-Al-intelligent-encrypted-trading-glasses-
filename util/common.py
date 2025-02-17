import os
import json
import logging
import datetime
import requests
from typing import List, Any, Dict


# ----------------------------
# File and JSON utility functions
# ----------------------------

def read_file(file_path: str, encoding: str = 'utf-8') -> str:
    """Read file content"""
    with open(file_path, 'r', encoding=encoding) as f:
        content = f.read()
    return content


def write_file(file_path: str, content: str, encoding: str = 'utf-8'):
    """Write content to file"""
    with open(file_path, 'w', encoding=encoding) as f:
        f.write(content)


def read_json(file_path: str) -> Dict:
    """Read JSON file and return a dictionary"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(data: Dict, file_path: str):
    """Write dictionary to a JSON file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ----------------------------
# List and string utility functions
# ----------------------------

def flatten_list(nested_list: List[List[Any]]) -> List[Any]:
    """Flatten a 2D list into a 1D list"""
    return [item for sublist in nested_list for item in sublist]


def remove_duplicates(lst: List[Any]) -> List[Any]:
    """Remove duplicate elements while preserving order"""
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def is_palindrome(s: str) -> bool:
    """Check if a string is a palindrome"""
    s = s.lower().replace(" ", "")
    return s == s[::-1]


# ----------------------------
# Time and date utility functions
# ----------------------------

def get_current_timestamp() -> str:
    """Get the current timestamp as a formatted string"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def parse_date(date_str: str, fmt: str = "%Y-%m-%d") -> datetime.datetime:
    """Parse a date string into a datetime object"""
    return datetime.datetime.strptime(date_str, fmt)


# ----------------------------
# Network utility functions
# ----------------------------

def download_file(url: str, save_path: str):
    """Download a file from a URL and save it locally"""
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded and saved to {save_path}")
    else:
        raise Exception(f"Download failed with status code {response.status_code}")


# ----------------------------
# Logger utility functions
# ----------------------------

def setup_logger(logger_name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with a file handler"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    # Create file handler and formatter
    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(fh)
    return logger


# ----------------------------
# Math utility functions
# ----------------------------

def factorial(n: int) -> int:
    """Calculate factorial recursively"""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    return 1 if n == 0 else n * factorial(n - 1)


def is_prime(n: int) -> bool:
    """Check if a number is prime"""
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True


# ----------------------------
# Example usage
# ----------------------------
if __name__ == "__main__":
    # File and JSON operations
    sample_text = "This is a sample text."
    write_file("sample.txt", sample_text)
    print("File content:", read_file("sample.txt"))

    sample_data = {"name": "Python Utilities", "version": 1.0}
    write_json(sample_data, "sample.json")
    print("JSON data:", read_json("sample.json"))

    # List and string utilities
    nested = [[1, 2], [3, 4, 2]]
    print("Flattened list:", flatten_list(nested))
    print("Unique elements:", remove_duplicates(flatten_list(nested)))
    print("Is 'radar' a palindrome?:", is_palindrome("radar"))

    # Time and date utilities
    print("Current timestamp:", get_current_timestamp())
    print("Parsed date:", parse_date("2025-02-17"))

    # Network utility: download file (ensure the URL is valid)
    # download_file("https://example.com/sample.jpg", "sample.jpg")

    # Logger utility
    logger = setup_logger("myLogger", "app.log")
    logger.info("This is a sample log entry.")

    # Math utilities
    print("Factorial of 5:", factorial(5))
    print("Is 7 prime?:", is_prime(7))
