import argparse
import heapq 
import pathlib

import numpy as np
import gymnasium as gym
from minigrid.core.constants import IDX_TO_OBJECT, OBJECT_TO_IDX 

import utils

def shortest_path(grid, pos, goal, include_goal=False):
    """
    Given a grid, solve the shortest path problem (Dijkstra) and 
    return a sequence of actions from current position and goal
    """

    # distance from start to each state
    pos_h = pos.tobytes()
    distance = {pos_h: 0}
    parent = {pos_h: (None, None)}
    states_ht = {pos_h: pos}

    OPEN = [(0, pos_h)]
    heapq.heapify(OPEN)

    CLOSED = set()

    moves = [np.array([1, 0]), np.array([0, 1]), np.array([-1, 0]), np.array([0, -1])]

    while len(OPEN) > 0:
        g, pos_h = heapq.heappop(OPEN)
        pos = states_ht[pos_h]

        # Check if goal is found
        if (pos == goal).all():
            break

        if pos_h in CLOSED:
            continue
        else:
            CLOSED.add(pos_h)


        for action, move in enumerate(moves):
            next_pos = pos + move

            # Check if next position is valid
            if next_pos[0] < 0 or next_pos[0] >= grid.shape[0]:
                continue
            if next_pos[1] < 0 or next_pos[1] >= grid.shape[1]:
                continue
            if grid[next_pos[0], next_pos[1]] == 2:
                continue

            next_pos_h = next_pos.tobytes()
            if next_pos_h not in states_ht:
                states_ht[next_pos_h] = next_pos

            # cost for moving to the next position
            if grid[next_pos[0], next_pos[1]] == 11:
                cost = 0.5
            elif grid[next_pos[0], next_pos[1]] == 9:
                cost = 2.0
            else:
                cost = 1.0

            tmp_dist = cost + distance[pos_h]
            if (next_pos_h not in CLOSED and 
                (next_pos_h not in distance or distance[next_pos_h] > tmp_dist)):
                distance[next_pos_h] = tmp_dist
                heapq.heappush(OPEN, (tmp_dist, next_pos_h))
                parent[next_pos_h] = (pos_h, action)
    path = retrieve_path(parent, states_ht, pos_h, include_goal=include_goal)
    return path


def retrieve_path(parent, states_ht, state_h, include_goal=False):
    """
    Retrieves a list of (state, action) from parent list to
    build the optimal path from start to goal 
    """
    if include_goal:
        path = [(states_ht[state_h], None)]
    else: 
        path = []
    while True:
        prev_state_h, prev_action = parent[state_h]
        if prev_state_h is None: break
        path.append((states_ht[prev_state_h], prev_action))
        state_h = prev_state_h
    return path[::-1]


def generate_episodes(env, save_dir, num):
    if not save_dir.exists():
        save_dir.mkdir(parents=True, exist_ok=False)

    for i in range(num):
        env.reset()   
        grid_map, goal = utils.extract_grid_map_and_goal(env.env.env.grid)
        pos = env.env.env.agent_pos

        episode = {
            'grid': env.env.env.grid.encode(),
            'grid_count': [],
            'lidar_pts': [], 
            'lidar_labels': [], 
            'agent_pos': [],
            'goal_pos': [],
            'action': []
        }

        path = shortest_path(grid_map, pos, goal, include_goal=True)   

        states = np.array([p[0] for p in path])
        actions = np.array([p[1] for p in path])    

        if not (states[0] == env.agent_pos).all() or not (states[-1] == goal).all():
            continue

        grid_count = np.zeros((*grid_map.shape, 4), dtype=np.int32)
        for state, action in zip(states[:-1], actions[:-1]):
            env.sem_lidar.set_pos(state)
            lidar_pts, lidar_labels = env.sem_lidar.detect()
            grid_count = utils.add_grid_counts(grid_count, lidar_pts, lidar_labels)
            episode['grid_count'].append(np.copy(grid_count))
            episode['lidar_pts'].append(np.copy(lidar_pts))
            episode['lidar_labels'].append(np.copy(lidar_labels))
            episode['agent_pos'].append(np.copy(state))
            episode['goal_pos'].append(np.copy(goal))
            episode['action'].append(np.copy(action))

        if utils.eplen(episode) > 0:
            utils.save_episode(save_dir, episode)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=16, const=16, nargs='?',
        choices=[16, 64], help='Minigrid map size')
    args = parser.parse_args()

    save_dir = pathlib.Path(f'./demonstrations/MiniGrid-LavaLawnS{args.grid_size}-v0')
    env = gym.make(
        f"MiniGrid-LavaLawnS{args.grid_size}-v0",
        tile_size=32,
        render_mode="human",
        screen_size=640,
        highlight=False,
        use_lidar=True,
    )

    num_train = 1000
    num_valid = 100
    generate_episodes(env, save_dir / 'train', num_train)
    generate_episodes(env, save_dir / 'valid', num_valid)


if __name__ == '__main__':
    main()


