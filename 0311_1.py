import tkinter as tk
from tkinter import ttk
import serial
import time

ser = serial.Serial("COM3",9600)

def on_button_click(args=None):
    ser.write("on".encode())
    status_str.set("on")
    #pass

def off_button_click(args=None):
    ser.write("off".encode())
    status_str.set("off")
    #pass

window = tk.Tk()
window.title("led button")
window.geometry("300x400")

on_button = ttk.Button(window, text="on", command=on_button_click)
on_button.pack(pady=20)

off_button = ttk.Button(window, text="off", command=off_button_click)
off_button.pack(pady=20)

status_str = tk.StringVar()
status_label = ttk.Label(window, text="off", textvariable=status_str)
status_label.pack(pady=20)

window.mainloop()

