import RPi.GPIO as GPIO
import time


if __name__ == '__main__':

    GPIO.setmode(GPIO.BCM)
    GPIO.setup(18, GPIO.OUT)

    while True:
        GPIO.output(18, GPIO.HIGH)
        time.sleep(2)
        GPIO.output(18, GPIO.LOW)
        time.sleep(2)
