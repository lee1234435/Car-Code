import tkinter as tk

def on_button_click():
    pass

def makeButton(root, title, bg_color, fg_color, w, h, com):
    button = tk.Button(root, text=title, bg=bg_color, fg=fg_color, width=w, height=h, command=com)
    # return button
    
# root 만들기
root = tk.Tk()
root.title("Buttons_GUI")
root.geometry("600x800")

# for color,side in [("red","left"), ("blue","right"), ("yellow","top"), ("green","bottom")]:    
#     button_final = makeButton(root, color, color, "white", 10, 10, on_button_click)
#     button_final.pack(side=side)

for color,side in [("red","left"), ("blue","right"), ("yellow","top"), ("green","bottom")]:    
    button_final = makeButton(root, color, color, "white", 10, 10, on_button_click)
    button_final.pack(side=side)

root.mainloop()



