import requests

import socket


def connect_to_device(host, port):
    # Create a TCP/IP socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Connect to the device
    s.connect((host, port))
    return s


def main():
    host = '192.168.1.100'  # Replace with the device IP address
    port = 12345  # Replace with the device port number

    # Establish connection to the device
    sock = connect_to_device(host, port)

    try:
        # Optionally send an initialization command to the device
        command = "GET_STATUS\n"
        sock.sendall(command.encode('utf-8'))

        # Receive data in a loop
        while True:
            data = sock.recv(4096)
            if not data:
                # No more data from the device
                break
            print("Received:", data.decode('utf-8'))
    except KeyboardInterrupt:
        print("Connection interrupted by user")
    finally:
        # Close the socket connection
        sock.close()


if __name__ == '__main__':
    main()

import socket
import select
import time
import logging
import threading

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')


class EyeDeviceClient:
    def __init__(self, host, port, timeout=5, reconnect_delay=5):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay
        self.socket = None
        self.connected = False
        self.stop_flag = False
        self.receive_thread = None

    def connect(self):
        """Establish a connection to the device."""
        while not self.connected and not self.stop_flag:
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.settimeout(self.timeout)
                self.socket.connect((self.host, self.port))
                self.socket.setblocking(False)
                self.connected = True
                logging.info(f"Connected to {self.host}:{self.port}")
            except Exception as e:
                logging.error(f"Connection failed: {e}")
                self.connected = False
                if self.socket:
                    self.socket.close()
                logging.info(f"Reconnecting in {self.reconnect_delay} seconds...")
                time.sleep(self.reconnect_delay)

    def disconnect(self):
        """Close the connection."""
        self.stop_flag = True
        if self.socket:
            self.socket.close()
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join()
        logging.info("Disconnected from device.")

    def send_command(self, command):
        """Send a command to the device."""
        if not self.connected:
            logging.warning("Not connected to the device. Cannot send command.")
            return
        try:
            self.socket.sendall(command.encode('utf-8'))
            logging.debug(f"Sent command: {command.strip()}")
        except Exception as e:
            logging.error(f"Error sending command: {e}")
            self.connected = False
            self.connect()

    def receive_loop(self):
        """Continuously receive data from the device."""
        while not self.stop_flag:
            if not self.connected:
                logging.info("Lost connection. Attempting to reconnect...")
                self.connect()
            try:
                # Use select to check if the socket has data
                ready_to_read, _, _ = select.select([self.socket], [], [], 1)
                if ready_to_read:
                    data = self.socket.recv(4096)
                    if data:
                        message = data.decode('utf-8', errors='replace')
                        logging.info(f"Received: {message.strip()}")
                    else:
                        logging.warning("No data received, connection may be closed by device.")
                        self.connected = False
                        self.socket.close()
                else:
                    time.sleep(0.1)
            except Exception as e:
                logging.error(f"Error in receive loop: {e}")
                self.connected = False
                if self.socket:
                    self.socket.close()

    def start_receiving(self):
        """Start the receiving thread."""
        if self.receive_thread is None or not self.receive_thread.is_alive():
            self.receive_thread = threading.Thread(target=self.receive_loop, daemon=True)
            self.receive_thread.start()
            logging.info("Started receive thread.")


def main():
    host = '192.168.1.100'  # Replace with actual device IP
    port = 12345  # Replace with actual device port
    client = EyeDeviceClient(host, port)

    try:
        client.connect()
        client.start_receiving()

        # Main loop to send commands periodically
        commands = [
            "GET_STATUS\n",
            "GET_DATA\n",
            "RESET\n"
        ]
        idx = 0
        while True:
            # Cycle through commands
            command = commands[idx % len(commands)]
            client.send_command(command)
            idx += 1
            time.sleep(3)  # wait between commands

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting.")
    finally:
        client.disconnect()


if __name__ == '__main__':
    main()


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
