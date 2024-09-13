# Orienteering

Welcome to the Orienteering project, a comprehensive tool designed to enhance the experience of orienteering athletes, course planners and event organizers. This project is an OpenSource alternative to OCAD path-finder, which is not as user-friendly due to costs and .ocd files. To satisfy the usability factor, this Path-finder (RouteAI) uses an image of the map in .PNG format. This applications is still in progress, but currently it brings together several key functionalities, including optimal route-choice testing, map conversion, and rogaining route calculations.

This project was done to learn about path-finding algorithms from Graph Theory. There are soo many improvements that could be made. In the longterm this app could be used by course setters to plan appropriately-long courses and predict finish times. For example, 30-35mins in WRE middle events.


# Features

The Orienteering Application offers three main features:

1. RouteAI
Description: Utilize advanced Image Processing algorithms and path-finding principles to see the optimal routes for orienteering controls. This feature enables athletes theoretically calculate the best route-choice for difficult controls.
Usage:
* Import a .PNG file to be analysed, choose "Forest or Sprint"
* Select the start and end control (press "Enter")
* Select an analysis boundary ("Drag and select")
* finally press "Enter" to begin analysis

3. MapConvert
Description: A tool for converting PNG orienteering map into a scattered grid of generalized pixels. This is used to see how the computer processes the image into a pixel array that it understands. Used for tweaking the image processing part of the RouteAI code.

4. Rogaining
Description: Plan your rogaining routes with ease. This app asks for the user input of all the controls, then using various path-finding algorithms to connect all the controls in a line.
Usage: Cheat at rogaining (not advised haha).
Improvements: Use Traveling Salesman Problem solutions, Minimun spanning tree etc., calculate the actual route.


# Installation

To install the Orienteering Application, follow these steps:

1. Clone the repository or download the source code to your local machine.
2. Instaling PIP: https://pip.pypa.io/en/stable/installation/
3. Ensure that Python (version 3.6 or later) is installed on your system.
4. Install required dependencies by running "pip3 install -r requirements.txt" in the terminal within the project directory.
5. Execute the main script to launch the application: "python main.py" OR launch it with Python Launcher.


# Usage

Upon launching the application, you will be greeted with a user-friendly interface displaying the three main features. Select the desired functionality to access its specific tools and options. 


# Customizing Routes and Maps

1. RouteAI: Open a PNG map file, enter the start point and end point for your desired route, then press "enter".
2. MapConvert: Upload the PNG map or a screenshot of part of the map.
3. Rogaining: Open a PNG file of the map. Choose the points, starting from the competition center. Press "enter".

For support, please open an issue on the GitHub repository page. We welcome feedback and suggestions for improving the application.

# Contributing

Contributions to the Orienteering Application are welcome. When sharing results of the app, please attribute the author "Jekabs Janovs".


# License

The Orienteering Application is released under the MIT Non-Commercial License. See the LICENSE file for more details.
