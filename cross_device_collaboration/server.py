import socket

# Set up a server to handle user interactions in the virtual trading community
def start_chat_server():
    host = '127.0.0.1'
    port = 65432

    # Create a socket to listen for connections
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, port))
        s.listen()

        print("Waiting for users to join the community...")
        conn, addr = s.accept()
        with conn:
            print(f"Connected by {addr}")
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                print(f"Message received: {data.decode()}")
                conn.sendall(data)  # Echo the message back to the user

# Start the server
start_chat_server()
