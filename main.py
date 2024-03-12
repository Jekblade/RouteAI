import tkinter as tk
import os
from turtle import back
from PIL import Image, ImageTk
import webbrowser

# Define root as a global variable
root = tk.Tk()
buttons = []

def route_ai():
    hide_main_buttons()
    show_route_ai_ui()

def map_convert():
    hide_main_buttons()
    show_map_convert_ui()

def rogaining():
    hide_main_buttons()
    show_rogaining_ui()


def hide_main_buttons():
    for button in buttons:
        button.place_forget()


def show_route_ai_ui():
    route_ai_label = tk.Label(root, text=" RouteAI ", font=('Roboto', 35))
    route_ai_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, y=-50)
    buttons.append(route_ai_label)

    # Create buttons 
    least_cost_button = tk.Button(root, text="LowestCostPath alg", font=('Roboto', 28), command=(execute_least_cost))
    least_cost_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(least_cost_button)

    djikstras_button = tk.Button(root, text="Djikstras alg", font=('Roboto', 28), command=(execute_djikstras))
    djikstras_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(djikstras_button)

    back_button = tk.Button(root, text="Back", font=('Roboto', 28), bg="gray", command=show_main_buttons)
    back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(back_button)

def execute_least_cost():
    os.system("python3 RouteAI/LeastCostPath.py")

def execute_djikstras():
    os.system("python3 RouteAI/DjikstrasAlg.py")


def show_map_convert_ui():
    map_convert_label = tk.Label(root, text=" Map convert ", font=('Roboto', 35))
    map_convert_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, y=-50)
    buttons.append(map_convert_label)

    map_convert_button = tk.Button(root, text="MapConvert", font=('Roboto', 28), command=(map_convert))
    map_convert_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(map_convert_button)

    terrain_button = tk.Button(root, text="Artificial Terrain", font=('Roboto', 28), command=(execute_lidar))
    terrain_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(terrain_button)

    back_button = tk.Button(root, text="Back", font=('Roboto', 28), bg="gray",  command=show_main_buttons)
    back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(back_button)

def execute_map_convert():
    os.system("python3 MapConvert/MapConvert.py")

def execute_lidar():
    os.system("python3 MapConvert/lidar.py")


def show_rogaining_ui():
    rogaining_label = tk.Label(root, text=" Rogaining ", font=('Roboto', 35))
    rogaining_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, y=-50)
    buttons.append(rogaining_label)

    nearest_neighbour_button = tk.Button(root, text="Nearest Neighbours", font=('Roboto', 28), command=(execute_nearest_neighbour))
    nearest_neighbour_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(nearest_neighbour_button)

    christofides_button = tk.Button(root, text="Christofides Method", font=('Roboto', 28), command=(execute_christofides))
    christofides_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(christofides_button)

    back_button = tk.Button(root, text="Back", font=('Roboto', 28), bg="gray", command=show_main_buttons)
    back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(back_button)

def execute_nearest_neighbour():
    os.system("python3 MapConvert/NearestNeighbour.py")

def execute_christofides():
    os.system("python3 MapConvert/Christofides.py")


def donate_paypal():
    webbrowser.open_new("https://www.paypal.com/donate/?hosted_button_id=SRNX7RECG8ZPQ")

# Homepage
def show_main_buttons():

    # 1
    hide_main_buttons()
    
    # 2
    choose_label = tk.Label(root, text=" Orienteering tools ", font=('Roboto', 35))
    choose_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, y=-50)
    buttons.append(choose_label)

    route_ai_button = tk.Button(root, text="RouteAI", font=('Roboto',28), command=route_ai)
    route_ai_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(route_ai_button)

    map_convert_button = tk.Button(root, text="MapConvert", font=('Roboto',28), command=map_convert)
    map_convert_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(map_convert_button)

    rogaining_button = tk.Button(root, text="Rogaining", font=('Roboto',28), command=rogaining)
    rogaining_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER,)
    buttons.append(rogaining_button)

    donate_button = tk.Button(root, text="Donate via PayPal", font=('Roboto',23), bg="yellow", fg="blue", command=donate_paypal)
    donate_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(donate_button)

def main():

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
    button_font = ('Roboto', 28) 
    button_padding = 60

    show_main_buttons()

    root.mainloop()

if __name__ == "__main__":
    main()
