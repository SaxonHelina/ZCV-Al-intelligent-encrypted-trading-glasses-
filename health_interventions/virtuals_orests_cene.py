import time
import numpy as np
import sounddevice as sd

# Placeholder function to simulate stress level detection
def get_stress_level():
    # In a real scenario, this function would interface with a wearable device
    # and return the current stress level as a numerical value
    return 8  # Simulated stress level on a scale from 1 to 10

# Define a stress threshold
STRESS_THRESHOLD = 7

import pygame

# Initialize Pygame
pygame.init()

# Set up the display
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("Virtual Forest Relaxation")

# Load the forest image
forest_image = pygame.image.load("forest_scene.jpg")

# Function to display the virtual forest scene
def display_forest_scene():
    screen.blit(forest_image, (0, 0))
    pygame.display.flip()

# Function to close the display
def close_forest_scene():
    pygame.quit()




# Function to generate and play a 4Hz sound wave
def play_infrasound(duration=10):
    fs = 44100  # Sampling frequency
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    wave = 0.5 * np.sin(2 * np.pi * 4 * t)  # 4Hz sine wave
    sd.play(wave, fs)
    sd.wait()


try:
    while True:
        stress_level = get_stress_level()
        if stress_level > STRESS_THRESHOLD:
            print("High stress detected. Activating relaxation protocol.")
            display_forest_scene()
            play_infrasound()
            time.sleep(10)  # Display duration
            close_forest_scene()
        time.sleep(5)  # Check stress level every 5 seconds
except KeyboardInterrupt:
    close_forest_scene()
    print("Program terminated.")
