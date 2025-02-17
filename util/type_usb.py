# Example 1: Connecting to a device via Serial using pySerial

import serial
import time

def connect_serial(port='/dev/ttyUSB0', baud_rate=9600):
    """Connect to a serial device."""
    try:
        ser = serial.Serial(port, baud_rate, timeout=1)
        print("Connected to serial port:", port)
        return ser
    except Exception as e:
        print("Error connecting to serial port:", e)
        return None

def read_serial_data(ser):
    """Read data from the serial device."""
    if ser is None:
        print("Serial connection not established.")
        return None
    try:
        line = ser.readline().decode('utf-8').rstrip()
        return line
    except Exception as e:
        print("Error reading from device:", e)
        return None

if __name__ == '__main__':
    serial_conn = connect_serial('/dev/ttyUSB0', 9600)
    while serial_conn:
        data = read_serial_data(serial_conn)
        if data:
            print("Received data:", data)
        time.sleep(1)


# Example 2: Connecting to an I2C device using smbus

import smbus

bus = smbus.SMBus(1)  # For Raspberry Pi, bus 1 is usually used
DEVICE_ADDRESS = 0x48  # Replace with your device's I2C address
REGISTER = 0x00        # Replace with the register you want to read

def read_i2c_sensor():
    """Read data from an I2C sensor."""
    try:
        data = bus.read_byte_data(DEVICE_ADDRESS, REGISTER)
        return data
    except Exception as e:
        print("Error reading I2C sensor:", e)
        return None

if __name__ == '__main__':
    while True:
        sensor_value = read_i2c_sensor()
        if sensor_value is not None:
            print("I2C Sensor Value:", sensor_value)
        time.sleep(1)


# Example 3: Controlling GPIO pins using RPi.GPIO

import RPi.GPIO as GPIO

LED_PIN = 18  # Replace with your GPIO pin number

def setup_gpio():
    """Set up the GPIO pins."""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)

def blink_led():
    """Blink an LED connected to the specified GPIO pin."""
    GPIO.output(LED_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(0.5)

if __name__ == '__main__':
    setup_gpio()
    try:
        while True:
            blink_led()
    except KeyboardInterrupt:
        print("Exiting program. Cleaning up GPIO settings.")
        GPIO.cleanup()
