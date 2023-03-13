import numpy as np
from map import map_score

box_true = np.array([
    [0, 15, 15, 22, 22, 1, 0],
    [0, 28, 28, 35, 35, 1, 0],
    [0, 40, 40, 54, 54, 1, 0],
    [0, 70, 70, 81, 81, 1, 0],
    [1, 35, 35, 32, 32, 1, 0],
    [1, 48, 48, 45, 45, 1, 0],
    [1, 70, 70, 81, 81, 1, 0],
    [1, 70, 70, 81, 81, 1, 0]
])
box_pred = np.array([
    [0.9, 15, 15, 25, 25],
    [0.7, 45, 45, 55, 55],
    [0.6, 50, 50, 60, 60],
    [0.5, 75, 75, 80, 80],
    [0.9, 45, 45, 25, 25],
    [0.7, 45, 45, 55, 55],
    [0.6, 50, 50, 60, 60],
    [0.5, 75, 75, 80, 80]
])

score = map_score(box_true, box_pred)
print(score)
