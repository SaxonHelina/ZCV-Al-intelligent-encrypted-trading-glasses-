import cv2
from deepface import DeepFace
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Emotion thresholds
emotion_thresholds = {
    'angry': 0.5,
    'disgust': 0.5,
    'fear': 0.5,
    'sad': 0.5
}

# Function to provide relaxation suggestions
def provide_suggestion(emotion):
    suggestions = {
        'angry': 'Take a deep breath and count to ten.',
        'disgust': 'Try to focus on something pleasant.',
        'fear': 'Remember, you are in a safe environment.',
        'sad': 'Consider listening to your favorite music.'
    }
    print(f"Suggestion: {suggestions.get(emotion, 'Stay calm and carry on.')}")
    # Play a relaxation sound
    playsound('relaxation_sound.mp3')

# Continuous emotion detection
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame for emotions
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Extract the dominant emotion and its confidence
    dominant_emotion = result['dominant_emotion']
    emotion_confidence = result['emotion'][dominant_emotion]

    # Check if the detected emotion exceeds the threshold
    if dominant_emotion in emotion_thresholds and emotion_confidence > emotion_thresholds[dominant_emotion]:
        print(f"Detected emotion: {dominant_emotion} with confidence {emotion_confidence}")
        provide_suggestion(dominant_emotion)
        time.sleep(5)  # Wait for 5 seconds before the next suggestion

    # Display the frame with the detected emotion
    cv2.putText(frame, f"{dominant_emotion}: {emotion_confidence:.2f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Emotion Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
