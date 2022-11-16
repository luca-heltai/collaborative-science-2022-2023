from interpolation import *

def test_cronecker():
    X = linspace(0,1,5)
    for i in range(len(X)):
        assert lagrange(X[i], i, X) == 1
        for j in range(len(X)):
            if i != j:
                assert lagrange(X[j], i, X) == 0

