import tkinter as tk
import os
from PIL import Image, ImageTk

def execute_route_ai():
    os.system("python RouteAI/RouteAI.py")

def execute_map_convert():
    os.system("python MapConvert/MapConvert.py")

def execute_rogaining():
    os.system("python Rogaining/Rogaining.py")

def main():
    # Create the main window
    root = tk.Tk()
    root.title("Orienteering")

    # Open the PNG file
    image_path = "files/background.png"
    background_image = Image.open(image_path)

    # Convert the image to Tkinter PhotoImage object
    background_photo = ImageTk.PhotoImage(background_image)

    # Get the image dimensions
    image_width, image_height = background_image.size

    # Set the window size and position based on the image dimensions
    window_width = image_width
    window_height = image_height
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_position = (screen_width - window_width) // 2
    y_position = (screen_height - window_height) // 2
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Display the background image
    background_label = tk.Label(root, image=background_photo)
    background_label.place(x=0, y=0, relwidth=1, relheight=1)

    # Define button styles
    button_font = ('Roboto', 29)  # Choose a modern font like Roboto
    button_width = 50
    button_height = 17
    button_padding = 60

    route_ai_button = tk.Button(root, text="RouteAI", font=button_font, command=execute_route_ai)
    route_ai_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER, y=-button_padding)

    map_convert_button = tk.Button(root, text="MapConvert", font=button_font, command=execute_map_convert)
    map_convert_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

    rogaining_button = tk.Button(root, text="Rogaining", font=button_font, command=execute_rogaining)
    rogaining_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER, y=button_padding)


    # Run the Tkinter event loop
    root.mainloop()

if __name__ == "__main__":
    main()
