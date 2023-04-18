
import numpy as np

import astar_pybind as pyAstar
import timeit
import os

# os.environ['OMP_NUM_THREADS'] = '4'

def test_astar():

    batch_size = 128
    num_controls = 4        # 0: +x , 1: +y, 2: -x, 3: -y
    grid_size = 16

    # constant cost of 1 for moving in any direction    
    cost = np.ones((batch_size, num_controls, grid_size, grid_size), dtype=np.float32)
    start = np.array([[1, 1] for i in range(batch_size)], dtype=np.int32)
    goal = np.array([[14, 14] for _ in range(batch_size)], dtype=np.int32)

    # Q(i, u) is the Q value from start[i] to goal[i] if first action is u
    Q = np.zeros((batch_size, num_controls), dtype=np.float32)

    # gradient of Q wrt cost
    dQdc = np.zeros((batch_size, num_controls, num_controls, 
                           grid_size, grid_size), dtype=np.float32)

    # g value for each state to goal 
    g = np.ones((batch_size, grid_size, grid_size), dtype=np.float32) * 1e6
    try:
        _ = os.environ["OMP_NUM_THREADS"]
    except KeyError:
        print("Please set OMP_NUM_THREADS before calling");
        raise

    print("Threads : %s; Time: " % os.environ["OMP_NUM_THREADS"],
        timeit.timeit(
            lambda: pyAstar.planBatch2DGrid(cost, start, goal, Q, dQdc, g),
            number=1000,
            globals=locals())
         )



if __name__ == '__main__':
    test_astar()
