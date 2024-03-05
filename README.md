# Orienteering

Welcome to the Orienteering project, a comprehensive tool designed to enhance the experience of orienteering athletes, course planners and event organizers. This versatile application brings together several key functionalities, including optimal route-choice testing, map conversion, and rogaining routes, all within a user-friendly interface.

This project was done to learn about path-finding algorithms from Graph Theory. There are soo many improvements that could be made. In the longterm this app could be used by course setters to plan appropriately-long courses and predict finish times. For example, 30-35mins in WRE middle events.


# Features

The Orienteering Application offers three main features:

1. RouteAI
Description: Utilize advanced Image Processing algorithms and path-finding principles to see the optimal routes for orienteering controls. This feature enables athletes theoretically calculate the best route-choice for difficult controls.
Usage: Ideal for event organizers and athletes to design courses or perform route-choice analysis.

2. MapConvert
Description: A tool for converting PNG orienteering map into a scattered grid of generalized pixels. This is used to see how the computer processes the image into a pixel array that it understands. Used for tweaking the image processing part of the RouteAI code.

3. Rogaining
Description: Plan your rogaining routes with ease. This app asks for the user input of all the controls, then using various path-finding algorithms to connect all the controls in a line.
Usage: Cheat at rogaining (not advised haha)


# Installation

To install the Orienteering Application, follow these steps:

1. Clone the repository or download the source code to your local machine.
2. Ensure that Python (version 3.6 or later) is installed on your system.
3. Install required dependencies by running "pip3 install -r requirements.txt" in the terminal within the project directory.
4. Execute the main script to launch the application: "python main.py" OR launch it with Python Launcher.


# Usage

Upon launching the application, you will be greeted with a user-friendly interface displaying the three main features. Select the desired functionality to access its specific tools and options. 


# Customizing Routes and Maps

1. RouteAI: Open a PNG map file, enter the start point and end point for your desired route, then press "enter".
2. MapConvert: Upload the PNG map or a screenshot of part of the map.
3. Rogaining: Open a PNG file of the map. Choose the points, starting from the competition center. Press "enter".

For support, please open an issue on the GitHub repository page. We welcome feedback and suggestions for improving the application.

# Contributing

Contributions to the Orienteering Application are welcome. Please refer to the CONTRIBUTING.md file for guidelines on submitting pull requests and participating in the development process.


# License

The Orienteering Application is released under the MIT License. See the LICENSE file for more details.
