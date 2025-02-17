import cv2
import mediapipe as mp
import speech_recognition as sr
import pyttsx3


class AIAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.commands = {
            "report": "Fetching financial report...",
            "strategy": "Adjusting trading strategy...",
            "exit": "Shutting down AI assistant."
        }

    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Capture and process user voice commands"""
        with sr.Microphone() as source:
            print("Listening for a command...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio).lower()
                return command
            except sr.UnknownValueError:
                return "Unrecognized command"
            except sr.RequestError:
                return "Voice service error"

    def detect_gesture(self, frame):
        """Detect hand gestures using Mediapipe"""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark

                if landmarks[8].y < landmarks[6].y and landmarks[12].y > landmarks[10].y:
                    return "thumbs_up"  # Gesture: Confirm action
                elif landmarks[8].x < landmarks[6].x and landmarks[12].x < landmarks[10].x:
                    return "swipe_left"  # Gesture: Switch mode
                elif landmarks[8].y > landmarks[6].y and landmarks[12].y > landmarks[10].y:
                    return "open_hand"  # Gesture: Stop AI assistant

        return None

    def start_interaction(self):
        """Start AI assistant with voice and gesture control"""
        cap = cv2.VideoCapture(0)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gesture = self.detect_gesture(frame)

            if gesture == "thumbs_up":
                self.speak("Confirmed action.")
            elif gesture == "swipe_left":
                self.speak("Switching mode.")
            elif gesture == "open_hand":
                self.speak("Stopping AI assistant.")
                break

            cv2.putText(frame, "Say a command or use a gesture", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("AI Assistant", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            voice_command = self.listen()
            if voice_command in self.commands:
                self.speak(self.commands[voice_command])
                if voice_command == "exit":
                    break

        cap.release()
        cv2.destroyAllWindows()


# Run the AI assistant
if __name__ == "__main__":
    assistant = AIAssistant()
    assistant.start_interaction()
