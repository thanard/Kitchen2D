import numpy as np
import time
from kitchen2d.multi_cups import MultiCups

env = MultiCups()
env.do_gui = True
N = 1
n_timesteps = 50
samples = env.sampled_x(N, n_timesteps)
x = list(samples)
for xx in x:
    start = time.time()
    env.setup(xx, n_timesteps)
    # print(time.time() - start)
    print(env.step(np.ones(3)))