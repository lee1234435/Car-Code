import serial

base_ang = 90
shoulder_ang = 90
elbow_ang = 90
wrist_ang = 90

cmd = f"a{base_ang}b{shoulder_ang}c{elbow_ang}d{wrist_ang}"

ser = serial.Serial("COM3",9600)

ser.write(cmd.encode())

