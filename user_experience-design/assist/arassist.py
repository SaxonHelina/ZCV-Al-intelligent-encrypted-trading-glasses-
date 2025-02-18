import cv2
import time
import pyttsx3
import speech_recognition as sr

# !/usr/bin/python3
# my own convert Dart version_in.cc to version.cc
#   to avoid the python version incompatible from running 'tools/make_version.py'
# but this script required snapshot hash as input
import os
import sys
import subprocess


def extract_tools_version(version_file):
    vals = {}
    with open(version_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            line = line.strip()
            if not line:
                continue
            k, v = line.split(' ', 1)
            vals[k] = v

    return vals


def get_short_git_hash():
    return subprocess.run(['git', 'rev-parse', '--short=10', 'HEAD'], capture_output=True,
                          check=True).stdout.decode().strip()


def get_git_timestamp():
    return subprocess.run(['git', 'log', '-n', '1', '--pretty=format:%cd'], capture_output=True,
                          check=True).stdout.decode().strip()


# sdk directory
SDK_DIR = sys.argv[1]
SNAPSHOT_HASH = sys.argv[2]
os.chdir(SDK_DIR)
SDK_DIR = '.'

# extract info from 'tools/VERSION'
tools_version_file = os.path.join(SDK_DIR, 'tools', 'VERSION')
version_info = extract_tools_version(tools_version_file)

version_info['SNAPSHOT_HASH'] = SNAPSHOT_HASH
version_info['GIT_HASH'] = get_short_git_hash()
version_info['COMMIT_TIME'] = get_git_timestamp()
# not same as Dart tools/make_version.py, but it is ok because it is just for displaying
version_info['VERSION_STR'] = f'{version_info["MAJOR"]}.{version_info["MINOR"]}.{version_info["PATCH"]}'

version_in_file = os.path.join(SDK_DIR, 'runtime', 'vm', 'version_in.cc')
version_out_file = os.path.join(SDK_DIR, 'runtime', 'vm', 'version.cc')

with open(version_in_file, 'r') as f:
    code = f.read()

for k, v in version_info.items():
    code = code.replace('{{' + k + '}}', v)

with open(version_out_file, 'w') as f:
    f.write(code)

class ARAssistant:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.recognizer = sr.Recognizer()
        self.assets_data = {
            "stocks": {"AAPL": 150.75, "TSLA": 705.50, "GOOGL": 2803.79},
            "crypto": {"BTC": 47000, "ETH": 3200}
        }

    def speak(self, text):
        """Convert text to speech"""
        self.engine.say(text)
        self.engine.runAndWait()

    def listen(self):
        """Capture user voice input"""
        with sr.Microphone() as source:
            print("Listening for commands...")
            try:
                audio = self.recognizer.listen(source, timeout=5)
                command = self.recognizer.recognize_google(audio)
                return command.lower()
            except sr.UnknownValueError:
                return "I didn't understand"
            except sr.RequestError:
                return "Speech service error"

    def show_ar_overlay(self, frame, text):
        """Display AR overlay with text"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    def get_asset_data(self, asset_type):
        """Retrieve asset data for AR display"""
        if asset_type in self.assets_data:
            return f"{asset_type.capitalize()} prices: {self.assets_data[asset_type]}"
        return "Unknown asset type"

    def start_ar_interface(self):
        """Start AR camera feed and AI interaction"""
        cap = cv2.VideoCapture(0)  # Open webcam

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            self.show_ar_overlay(frame, "Say 'check stocks' or 'check crypto'")

            cv2.imshow("AR Assistant", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            command = self.listen()
            if "stocks" in command:
                message = self.get_asset_data("stocks")
            elif "crypto" in command:
                message = self.get_asset_data("crypto")
            else:
                message = "Invalid command"

            self.speak(message)

        cap.release()
        cv2.destroyAllWindows()


# Start the AR assistant
if __name__ == "__main__":
    assistant = ARAssistant()
    assistant.start_ar_interface()
