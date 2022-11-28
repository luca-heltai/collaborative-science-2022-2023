from universityTimetable import *

def testZeroElements():
    """
    Here the total cost of overlaps should obviously be zero
    """
    # Courses list
    coursesList = [
    ]

    # Overlaps that MUST be avoided
    mustAvoidList = [
    ]

    # Overlaps that SHOULD be avoided
    shouldAvoidList = [
    ]

    solver = Solver()
    solver.solve(coursesList, mustAvoidList, shouldAvoidList, printSolution=False)

    assert int(solver.computeCost()) == 0

def testTwelveElements():
    """
    In this case some overlaps are unavoidable, but it's possible to only have "unimportant" overlaps (each of cost 1)
    """
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

    solver = Solver()
    solver.solve(coursesList, mustAvoidList, shouldAvoidList, printSolution=False)

    assert int(solver.computeCost()) == 4

def testRareCases():
    """
    Same as before, but the cost of an overlap is always 3, except for some rare cases in which it is 1.
    The algorithm should be able to find those rare cases.
    """
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
    ]

    # Overlaps that SHOULD be avoided
    shouldAvoidList = [
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    ]

    solver = Solver()
    solver.solve(coursesList, mustAvoidList, shouldAvoidList, printSolution=False)

    assert int(solver.computeCost()) == 4

def testAlwaysOneHundred():
    """
    Here all overlaps must be avoided, so the cost of an overlap is always 100
    """
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
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    ]

    # Overlaps that SHOULD be avoided
    shouldAvoidList = [
    ]

    solver = Solver()
    solver.solve(coursesList, mustAvoidList, shouldAvoidList, printSolution=False)

    assert int(solver.computeCost()) == 400