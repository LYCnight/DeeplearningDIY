# conda activate marker

import numpy as np
t = np.arange(12).reshape(3,4)
print(t)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]]

t1 = np.sum(t, axis = 0) # size:[1, 4]
print(t1)   # [12 15 18 21]

t2 = np.sum(t, axis = 1) # size:[1, 3]
print(t2)   # [ 6 22 38]
