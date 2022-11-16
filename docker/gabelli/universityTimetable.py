import pandas as pd
import numpy as np
from pyomo.environ import *

class Course:
  def __init__(self, name, lecturesPerWeek):
    self.name = name # Name of the course
    self.lecturesPerWeek = lecturesPerWeek # Numbers of lectures per week

# Courses list
coursesList = [
    Course("Advanced Topics in Machine Learning", 2),
    Course("Cyber-Physical Systems", 2),
    Course("Natural Language Processing", 2),
    Course("Control Theory", 2),
    Course("Unsupervised Learning", 2),
    Course("Stochastic Modelling and Simulation", 2),
    Course("Mathematical Optimization", 2),
    Course("Bayesian Statistics", 2),
    Course("Statistical Learning for Data Science", 2),
    Course("Advanced Data Management and Curation", 2),
    Course("Introduction to Quantum Information Theory", 2),
    Course("Advanced Quantum Computing", 2)
    ]

# Overlaps that MUST be avoided
mustAvoidList = [
    [0, 2, 4],
    [7, 8, 9],
    [1, 10, 11]
    ]

# Overlaps that SHOULD be avoided
shouldAvoidList = [
    [6, 3, 11]
    ]

model = ConcreteModel()
model.I = RangeSet(1, len(coursesList))
model.J = RangeSet(1, len(coursesList))
model.H = RangeSet(1, 4)
model.K = RangeSet(1, 5)

model.x = Var(model.I, model.H, model.K, domain = Binary)
model.y = Var(model.I, model.I, model.H, model.K, domain = Binary)

c = np.ones((len(coursesList), len(coursesList)))
for k, l in enumerate(mustAvoidList):
  for i in l:
    for j in l:
      if i < j: # We only consider i < j
        c[i][j] = max(100 - k/len(mustAvoidList), c[i][j]) # The containers that come first have higher priority

for k, l in enumerate(shouldAvoidList):
  for i in l:
    for j in l:
      if i < j: # We only consider i < j
        c[i][j] = max(3 - k/len(shouldAvoidList), c[i][j]) # The containers that come first have higher priority

# Every course should appear the right number of times
model.meaningOfX = ConstraintList()
for i in model.I:
    model.meaningOfX.add(sum(model.x[i,h,k] for h in model.H for k in model.K) == coursesList[i-1].lecturesPerWeek)

# Count the cost of overlaps
model.meaningOfY = ConstraintList()
for h in model.H:
    for k in model.K:
        for i in model.I:
            for j in model.I:
                if i != j:
                    model.meaningOfY.add(model.y[i,j,h,k] >= model.x[i,h,k]+model.x[j,h,k]-1)
                    model.meaningOfY.add(model.y[i,j,h,k] <= model.x[i,h,k])
                    model.meaningOfY.add(model.y[i,j,h,k] <= model.x[j,h,k])
            model.meaningOfY.add(model.y[i,i,h,k] == 0)

# Avoid having the same course twice in a day
model.sameDay = ConstraintList()
for i in model.I:
  for k in model.K:
    model.sameDay.add(sum(model.x[i,h,k] for h in model.H) <= 1)

model.obj = Objective(expr = sum(model.y[i,j,h,k] * c[i-1][j-1] for i in model.I for j in model.I if j > i for h in model.H for k in model.K), sense = minimize)

# Solve the model
# SOLVER_NAME = 'gurobi'
SOLVER_NAME = 'glpk'

solver = SolverFactory(SOLVER_NAME)
TIME_LIMIT = max(5, len(coursesList) // 2) # You might want to change this

if SOLVER_NAME == 'glpk':
    solver.options['tmlim'] = TIME_LIMIT
elif SOLVER_NAME == 'gurobi':
    solver.options['TimeLimit'] = TIME_LIMIT

sol = solver.solve(model, tee=True)

table = {time: {day: [] for day in range(5)} for time in range(4)}
for i in model.I:
    for h in model.H:
        for k in model.K:
            if model.x[i,h,k]() == 1:
                table[h-1][k-1].append(coursesList[i-1].name)
df = pd.DataFrame(table)
df.index = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
df.columns = ["9-11", "11-13", "14-16", "16-18"]

print("Printing data to out.csv")
df.to_csv("out.csv", encoding="utf-8")
print("Done")