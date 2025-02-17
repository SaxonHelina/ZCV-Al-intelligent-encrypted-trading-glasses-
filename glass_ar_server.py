from flask import Flask, request
import os

app = Flask(__name__)

# Directory to store received images
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Set up the API route to handle image uploads
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return 'No image part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded image
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    return f'Image uploaded successfully: {file.filename}', 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
