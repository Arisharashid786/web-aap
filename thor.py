import tkinter as tk
from tkinter import messagebox


def on_button_click():
    name = entry.get()
    if name:
        messagebox.showinfo("Greeting", f"Hello, {name}!")
    else:
        messagebox.showwarning("Warning", "Please enter your name!")


# Create the main window
root = tk.Tk()
root.title("My First Python App")
root.geometry("300x200")

# Add a label
label = tk.Label(root, text="Enter your name:", font=("Arial", 12))
label.pack(pady=10)

# Add a text entry box
entry = tk.Entry(root, font=("Arial", 12))
entry.pack(pady=5)

# Add a button
button = tk.Button(
    root, text="Click Me", command=on_button_click, font=("Arial", 11)
)
button.pack(pady=15)

# Run the application loop
root.mainloop()