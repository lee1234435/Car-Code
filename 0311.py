# 파이썬으로 아두이노 제어
import serial
import time

ser = serial.Serial("COM3",9600)

while True:
    command = input("명령어를 넣어주세요.")

    ser.write(command.encode())
    time.sleep(0.5)


