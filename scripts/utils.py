import datetime
import uuid
import io
import pathlib
import numpy as np

labels = {
    1: 0,       # empty
    2: 1,       # wall
    8: 0,       # goal
    9: 2,       # lava
    11: 3,      # lawn
}

def save_episode(directory, episode):

    # Pad lidar points with 

    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    identifier = str(uuid.uuid4().hex)
    length = eplen(episode)
    filename = directory / f'{timestamp}-{identifier}-{length}.npz'
    with io.BytesIO() as f1:
        np.savez_compressed(f1, **episode)
        f1.seek(0)
        with filename.open('wb') as f2:
            f2.write(f1.read())
    return filename


def load_episodes(directory, capacity=None):
    # The returned directory from filenames to episodes is guaranteed to be in
    # temporally sorted order.
    filenames = sorted(directory.glob('*.npz'))
    if capacity:
        num_steps = 0
        num_episodes = 0
        for filename in reversed(filenames):
            length = int(str(filename).split('-')[-1][:-4])
            num_steps += length
            num_episodes += 1
            if num_steps >= capacity:
                break
        filenames = filenames[-num_episodes:]
    episodes = {}
    for filename in filenames:
        try:
            with filename.open('rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}
        except Exception as e:
            print(f'Could not load episode {str(filename)}: {e}')
            continue
        episodes[str(filename)] = episode
    return episodes

def eplen(episode):
    return len(episode['action'])

def add_grid_counts(grid_count, lidar_pts, lidar_labels):
    """
    Add lidar points to semantic counts on grid
    """
    grid_idx = np.ceil(lidar_pts - 0.5).astype(int)
    for idx, l in zip(grid_idx, lidar_labels):
        grid_count[idx[0], idx[1], labels[l]] += 1
    return grid_count

def extract_grid_map_and_goal(grid):
    grid_map = grid.encode()[:,:,0]
    goal_loc = np.where(grid_map == 8)
    return grid_map, np.concatenate(goal_loc)
