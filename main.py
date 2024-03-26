import tkinter as tk
import os
from turtle import back
from PIL import Image, ImageTk
import webbrowser

# Define root as a global variable
root = tk.Tk()
buttons = []

# On hover:
class HoverButton(tk.Button):
    def __init__(self, master=None, **kw):
        tk.Button.__init__(self, master=master, **kw)
        self.default_bg = self["background"]
        self.bind("<Enter>", self.on_enter)
        self.bind("<Leave>", self.on_leave)

    def on_enter(self, event):
        self["background"] = "light blue"
        self.show_image()

    def on_leave(self, event):
        self["background"] = self.default_bg
        self.hide_image()

    def show_image(self):
        button_name = self.cget("text")
        image_path = f"files/images/{button_name}.png"
        if os.path.exists(image_path):
            self.image = Image.open(image_path)
            self.image = self.image.resize((150, 150))
            self.image = ImageTk.PhotoImage(self.image)
            self.image_label = tk.Label(root, image=self.image)
            self.image_label.place(relx=1.1, rely=0.5, anchor='center')

    def hide_image(self):
        if hasattr(self, 'image_label'):
            self.image_label.destroy()

#Buttons
def route_ai():
    hide_main_buttons()
    show_route_ai_ui()

def map_convert():
    os.system('python3 MapConvert/mapConvert.py')

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
    djikstras_button = HoverButton(root, text="Djikstras Alg (pixel by pixel)", font=('Roboto', 28), command=(execute_djikstras))
    djikstras_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(djikstras_button)

    least_cost_button = HoverButton(root, text="LowestCostPath (manual approach)", font=('Roboto', 28), command=(execute_least_cost))
    least_cost_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(least_cost_button)

    djikstras_area_button = HoverButton(root, text="RouteAI (Djikstras+Area)", font=('Roboto', 28), command=(execute_djikstras_area))
    djikstras_area_button.place(relx=0.5, rely=0.6, anchor=tk.CENTER)
    buttons.append(djikstras_area_button)

    back_button = tk.Button(root, text="Back", font=('Roboto', 28), bg="gray", command=show_main_buttons)
    back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(back_button)

def execute_djikstras():
    os.system("python3 RouteAI/DjikstrasAlg.py")

def execute_least_cost():
    os.system("python3 RouteAI/LeastCostPath.py")

def execute_djikstras_area():
    os.system("python3 RouteAI/DjikstrasAREA.py")



def show_rogaining_ui():
    rogaining_label = tk.Label(root, text=" Rogaining ", font=('Roboto', 35))
    rogaining_label.place(relx=0.5, rely=0.3, anchor=tk.CENTER, y=-50)
    buttons.append(rogaining_label)

    nearest_neighbour_button = HoverButton(root, text="Nearest Neighbours", font=('Roboto', 28), command=(execute_nearest_neighbour))
    nearest_neighbour_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(nearest_neighbour_button)

    christofides_button = HoverButton(root, text="Christofides Method", font=('Roboto', 28), command=(execute_christofides))
    christofides_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(christofides_button)

    back_button = tk.Button(root, text="Back", font=('Roboto', 28), bg="gray", command=show_main_buttons)
    back_button.place(relx=0.5, rely=0.8, anchor=tk.CENTER, y=60)
    buttons.append(back_button)

def execute_nearest_neighbour():
    os.system("python3 Rogaining/NearestNeighbour.py")

def execute_christofides():
    os.system("python3 Rogaining/Christofides.py")


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

    route_ai_button = HoverButton(root, text="RouteAI", font=('Roboto',28), command=route_ai)
    route_ai_button.place(relx=0.5, rely=0.4, anchor=tk.CENTER)
    buttons.append(route_ai_button)

    map_convert_button = HoverButton(root, text="MapConvert", font=('Roboto',28), command=map_convert)
    map_convert_button.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
    buttons.append(map_convert_button)

    rogaining_button = HoverButton(root, text="Rogaining", font=('Roboto',28), command=rogaining)
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

    show_main_buttons()

    root.mainloop()

if __name__ == "__main__":
    main()
