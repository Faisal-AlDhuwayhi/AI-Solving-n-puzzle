# Artificial Intelligence - Solving N-puzzle Problem

## 1 Objective
**The goal of the project** is to build an artificial intelligent that can solve n-puzzle problem using informed and uninformed search methods.

## 2 Requirements
1. Implement the n-puzzle problem.
1. Your tool should accept n as input, along with the initial state. Initial state could be randomly generated as well.
1. Your tool allows user to select search strategy as input (drop down list).
1. Solve the problem using the selected search strategy.

## 3 The Problem
Given a n×n board with n^2 tiles (every tile has one number from 1 to n^2-1) and one
empty space. The objective is to place the numbers on tiles to match final configuration
using the empty space. You can slide four adjacent tiles (left, right, down and up) into
the empty space. Figure below shows the target when n=4. Furthuermore, each state
will be represented as 2D array (array\[n]\[n]), and the transtion between states will be
through moving the empty square, which will have the value of 0. 

![Figure demonstrates the problem when n=4](https://algorithmsinsight.files.wordpress.com/2016/03/220px-15-puzzle-svg.png?w=730)

You can find more details about this project in [**the report file**](https://github.com/Faisal-AlDhuwayhi/AI-Solving-n-puzzle/blob/master/Report.pdf).

## 4 Output
You should produce these **Outputs** at the end of the project:
1. Total number of steps to reach solution (path cost).
1. Total number of processed nodes until the strategy finds solution.
1. Maximum number of nodes that have been stored concurrently.
1. Simulation playback of the solution process showing the transitions from the initial state to the goal state. (time delay should be accepted as input)

[**The batch file**](https://github.com/Faisal-AlDhuwayhi/AI-Solving-n-puzzle/blob/master/Batch_file.xlsx) above expresses some statistics and graphs that show and compare the results between the different algorithms. Take into concern that the results are relative to the user machine.  

## 5 Using the code of the project
The code of the project contains third-party library of python like `numpy`. so you need to install it in your machine and then start working.

last thing to mention is that, it's better to include [venv](/venv) (vitual enviroment) folder when installing the project. because it has all the settings of the project you need.

You can find the plain code at [n-puzzle file](n-puzzle.py).
