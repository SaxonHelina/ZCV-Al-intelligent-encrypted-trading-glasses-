import requests
import os

# The server's API URL
SERVER_URL = "http://127.0.0.1:5000/upload-image"

# Path to the image (can be a captured image from the wearable glasses camera)
image_path = 'test_image.jpg'

# Check if the image exists
if not os.path.exists(image_path):
    print(f"Error: The file {image_path} does not exist.")
else:
    with open(image_path, 'rb') as img_file:
        files = {'image': (image_path, img_file, 'image/jpeg')}

        # Send a POST request with the image file
        response = requests.post(SERVER_URL, files=files)

        # Check if the upload was successful
        if response.status_code == 200:
            print('Image successfully uploaded to the server.')
        else:
            print(f"Failed to upload image. Server responded with: {response.status_code}")
