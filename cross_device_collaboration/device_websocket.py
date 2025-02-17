import asyncio
import websockets
import json

# This function will be called when the user performs an action on one device
async def device_action(websocket, path):
    data = await websocket.recv()  # Get action data (like trade or strategy)
    action = json.loads(data)
    print(f"Action received: {action}")

    # Broadcast the action to other devices
    await websocket.send(json.dumps(action))

# Start the WebSocket server to sync data across devices
async def start_server():
    server = await websockets.serve(device_action, "localhost", 8765)
    await server.wait_closed()

# Start the server
asyncio.run(start_server())
