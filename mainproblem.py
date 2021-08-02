from typing import List, Any
import math
import problem_solution
import os
import cv2 as cv
import re
import numpy as np
import copy
np.seterr(over='ignore') # ignore RuntimeWarning: overflow encountered in ubyte_scalars


class Map:
    def __init__(self, pathway):
        self.data = cv.imread(os.path.join(pathway), 0)

    def set(self, pos, value):
        self.data[pos[0], pos[1]] = value

    def all_move(self, pos):
        # all move can perform from the position pos
        # return action and new position
        ans = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                newPos: tuple[int, int] = (pos[0] + i, pos[1] + j)
                if (not pos == newPos) and (0 <= newPos[0] < self.data.shape[0]) and (0 <= newPos[1] < self.data.shape[1]):
                    ans.append(("move from" + str(pos) + "to" + str(newPos), newPos))
        return ans

class PathWaySearchProblem(problem_solution.SearchProblem):
    def cleanScreen(self):
        self.count_show = 0
        self.showmap = copy.copy(self.map.data)
    def __init__(self, data_pathway, in_pathway):
        self.count_show = 0
        self.map = Map(data_pathway)
        self.showmap = copy.copy(self.map.data)
        f = open(in_pathway)
        self.startPos = f.read
        # open and read input data
        f = open("input.txt")
        a = f.read()
        f.close()
        # input is a array contain all data number
        data_in = [int(x.group()) for x in re.finditer(r'\d+', a)]
        # assign data to init state and goal state
        self.init = (data_in[1], data_in[0])
        self.goal = (data_in[3], data_in[2])
        self.m = data_in[4]
    def display_explored(self,state):
        self.showmap[state[0]][state[1]]= 255
        self.count_show = self.count_show+1
        if self.count_show % 100 ==0:
            cv.imshow('img1',self.showmap)
            cv.waitKey(1)
    def initState(self):
        return self.init

    def isGoal(self, state):
        return self.goal == state
    def heuristic(self,state:tuple[int,int],heuristic_choice):
        delta_a = int(self.map.data[state[0]][state[1]]) - int(self.map.data[self.goal[0]][self.goal[1]])
        switcher = {
            1 : (math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2)),
            2 : (math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2) + \
                 + 0.5 * math.fabs(delta_a)),
            3: (math.sqrt((state[0]-self.goal[0])**2 + (state[1]-self.goal[1])**2) + \
                 + (0.5*np.sign(delta_a)+1) * math.fabs(delta_a))
        }
        return switcher.get(heuristic_choice)
    def expandSuccessor(self, state):
        ans = []
        for action, newstate in self.map.all_move(state):
            delta_a = int(self.map.data[state[0]][state[1]]) - int(self.map.data[newstate[0]][newstate[1]])
            if(math.fabs(delta_a) > self.m): continue
            addcost = math.sqrt((state[0]-newstate[0])**2+(state[1]-newstate[1])**2) \
                      + (0.5 * np.sign(delta_a)+1)*math.fabs(delta_a)
            ans.append((action,newstate,addcost))
        return ans


# DEBUG_SESSION : class Map
# test = Map("Map.bmp")
# for action, newPos in (test.all_move((47, 192))):
#     print(test.data[newPos])
# cv.imshow('image',test.data)
#
# test.set((213,74),255)
# test.set((311,96),255)
# test.set((96,311),255)
# cv.imshow('image',test.data)
# cv.waitKey(0)
# END DEBUG

# TEST_SESSION : read start - end position from input.txt
# f = open("input.txt")
# a = f.read()
# input = [int(x.group()) for x in re.finditer(r'\d+', a)]
# print(input)

#DEBUG SESSION : test class PathWaySearchProblem
problem = PathWaySearchProblem("map.bmp", "input.txt")
#heuristic 1
print("heuristic 1 : \n")
sovler1 = problem_solution.A_StarSolution()
sovler1.setHeuristic(1)
sovler1.solve(problem, 1)
newmap1 = Map("map.bmp")
for action, reached_state in sovler1.actions:
    newmap1.set(reached_state,255)
cv.imwrite("map1.bmp", newmap1.data)
problem.cleanScreen()
# #heuristic 2
print("#heuristic 2 :")

sovler2 = problem_solution.A_StarSolution()
sovler2.setHeuristic(2)
sovler2.solve(problem, 1)
newmap2= Map("map.bmp")
for action, reached_state in sovler2.actions:
    newmap2.set(reached_state,255)
cv.imwrite("map2.bmp", newmap2.data)
problem.cleanScreen()
#heuristic 3
print("#heuristic 3 :")
sovler2.setHeuristic(3)
sovler2.solve(problem,1)
newmap3 = Map("map.bmp")
for action, reached_state in sovler2.actions:
    newmap3.set(reached_state,255)
cv.imwrite("map3.bmp", newmap3.data)
problem.cleanScreen()
#UCS
print("#UCS algo :")
sovler3_UCS = problem_solution.UniformSolution()
sovler3_UCS.solve(problem, 1)
newmap3= Map("map.bmp")
for action, reached_state in sovler3_UCS.actions:
    newmap3.set(reached_state,255)
cv.imwrite("map3_optimal.bmp", newmap3.data)
#END SESION