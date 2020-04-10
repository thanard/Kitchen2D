import numpy as np
import h5py
import time
import matplotlib.pyplot as plt
from kitchen2d.multi_cups import MultiCups
from joblib import Parallel, delayed

def save_raw_image(file, image):
  plt.imshow(image)
  with open(file, 'wb') as sp:
    plt.savefig(sp)

# Arguments
n_tasks = 1000
n_timesteps = 100
height = width = 64
action_dim = 3
datapath = 'data/pouring_3_cups_small_pixel_translation_repeat_actions.hdf5'

# Set up files
f = h5py.File(datapath, 'w')
sim_data = f.create_group('sim')
sim_data.create_dataset(
  'ims', (n_tasks, n_timesteps, height, width, 3), dtype='f')
sim_data.create_dataset('actions', (n_tasks, n_timesteps, action_dim), dtype='f')
# sim_data.create_dataset('states', (forward_ep, steps_per_ep, 4), dtype='f')

# Parallel DataCollection
n_jobs = 50
n_tasks_per_job = 20
def gen_traj(parallel_id):
    if parallel_id == 0:
        start = time.time()
    env = MultiCups()
    env.do_gui = True
    all_tasks = list(env.sampled_x(n_tasks_per_job))
    observations = np.zeros((n_tasks_per_job, n_timesteps, height, width, 3))
    actions = np.zeros((n_tasks_per_job, n_timesteps, action_dim))
    for task_id, each_task in enumerate(all_tasks):
        print("### Thread %d: Task %d ###" % (parallel_id, task_id))
        env.setup(each_task)
        # print(time.time() - start)
        # action = np.array([0., 0., 0.])
        obs = env.render()
        action_repeats = 0
        for timestep in range(n_timesteps):
            if action_repeats == 0:
                mode = np.random.choice(['rotate','translate'], p=[0.4, 0.6])
                action_repeats = np.random.choice([10, 15, 20])
                action = np.random.uniform(-0.2, 0.2, size=3)
                if mode == 'rotate':
                    action[:2] = 0.
                    action[2] = np.random.choice([-0.05, 0.05])
                else:
                    action[2] = 0.
            else:
                action_repeats -= 1
            observations[task_id, timestep] = obs
            actions[task_id, timestep] = action
            env.step(action)
            obs = env.render()
    if parallel_id == 0:
        print(time.time()-start)
    return observations, actions
x = Parallel(n_jobs=n_jobs, backend='multiprocessing')(
    delayed(gen_traj)(i) for i in range(n_jobs))
for job_id, (observations, actions) in enumerate(x):
    start_idx = job_id * n_tasks_per_job
    end_idx = (job_id + 1) * n_tasks_per_job
    f['sim']['ims'][start_idx: end_idx] = observations
    f['sim']['actions'][start_idx: end_idx] = actions
f.flush()
f.close()
import ipdb
ipdb.set_trace()

# Single Thread
env = MultiCups()
env.do_gui = True
all_tasks = list(env.sampled_x(n_tasks))
for task_id, each_task in enumerate(all_tasks):
    print("### Task %d ###" % task_id)
    env.setup(each_task)
    # print(time.time() - start)
    # action = np.array([0., 0., 0.])
    obs = env.render()
    action_repeats = 0
    for timestep in range(n_timesteps):
        if action_repeats == 0:
            mode = np.random.choice(['rotate','translate'], p=[0.7, 0.3])
            action_repeats = np.random.choice([10, 15, 20])
            action = np.random.uniform(-0.2, 0.2, size=3)
            if mode == 'rotate':
                action[:2] = 0.
                action[2] = np.random.choice([-0.05, 0.05])
            else:
                action[2] = 0.
        else:
            action_repeats -= 1

    	f['sim']['ims'][task_id, timestep] = obs
        f['sim']['actions'][task_id, timestep] = action
        print(env.step(action))
        obs = env.render()
        # env.gripper.compute_post_grasp_mass()
        # print(env.gripper.mass)
        # env.render()
        if task_id == 0:
            env.save_observation('tmp_images/%05d_test.png' % timestep)
            save_raw_image('tmp_images/low_res_%05d.png' % timestep, obs)

        # import ipdb
        # ipdb.set_trace()
        env.gripper.compute_post_grasp_mass()
        print(env.gripper.mass)
f.flush()
f.close()

