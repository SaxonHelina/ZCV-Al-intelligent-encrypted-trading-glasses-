import cv2
import numpy as np

def capture_iris_image():
    """Capture an image of the eye using the default camera."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None

    print("Please position your eye in front of the camera and press 'c' to capture.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        cv2.imshow('Capture Eye Image', frame)
        if cv2.waitKey(1) & 0xFF == ord('c'):
            eye_image = frame
            break

    cap.release()
    cv2.destroyAllWindows()
    return eye_image

def preprocess_image(image):
    """Convert the image to grayscale and apply Gaussian blur."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (7, 7), 0)
    return blurred_image

def detect_iris(image):
    """Detect the iris in the image using HoughCircles."""
    circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=50, param2=30, minRadius=20, maxRadius=80)
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
        return circles[0]
    else:
        print("Iris not detected.")
        return None

def extract_iris_features(image, circle):
    """Extract features from the detected iris region."""
    x, y, r = circle
    iris_region = image[y-r:y+r, x-r:x+r]
    resized_iris = cv2.resize(iris_region, (64, 64))
    features = resized_iris.flatten()
    return features

def save_iris_features(features, filename):
    """Save the extracted iris features to a file."""
    np.save(filename, features)
    print(f"Iris features saved to {filename}")

if __name__ == "__main__":
    eye_image = capture_iris_image()
    if eye_image is not None:
        preprocessed_image = preprocess_image(eye_image)
        iris_circle = detect_iris(preprocessed_image)
        if iris_circle is not None:
            iris_features = extract_iris_features(preprocessed_image, iris_circle)
            save_iris_features(iris_features, "stored_iris_features.npy")
