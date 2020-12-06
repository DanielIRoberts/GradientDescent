# GradientDescent
A gradient descent algorithm that impliments Adagrad.

General Info---------------------------------------------------------------------------------------------

Name	- Daniel Roberts	NetID	- dir170130
	- Daoqun Yang			- dxy170330

Assignment: Assignment 1 (Enhanced Gradient Descent)
IDE	- Visual Studio Code
Python Version - 3.7.2

Included Files-------------------------------------------------------------------------------------------

Gradientdescent.py
	-Reading dataset source form URL and preprocessing the data
	-Randomlization of the weigh, generated through a normal distribution at the beginning
	- Calculating the Error, Mean Square Error (MSE), and derivative of MSE
	- Updating the weight using Adaptive Gradient Descent formula until max iterations are reached
	  or a minimum change is met
	- Accepts max iterations, learning rate, beta (for adagrad), and a plot name
	- Program will prompt for hyper variables
	- Using matplotlib for visualization


How to Run-----------------------------------------------------------------------------------------------

Both parts of the code can be run either directly through the .py file or through command line. Part 1 
requires some input that can be done through command line or when prompted shown in the example that 
follows.

Example Input Text---------------------------------------------------------------------------------------

If prompted:
Please enter the max iterations: 100
Please enter desired alpha: 0.1
Please enter desired beta:: 0.9
Enter name of the plot: plotName
