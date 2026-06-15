'''
python3 UMBC/experimental/generate_csv.py
'''

import glob
import pandas as pd

def merge_csv(observation, action, batch_size=32, input_dim=257): # zero padding
    observation_df = pd.DataFrame()
    action_df = pd.DataFrame()
    for pair in zip(observation, action):
      obs_df = pd.read_csv(pair[0]) # ex: obs
      act_df = pd.read_csv(pair[1]) # ex: action
      # print(pair)
      print(obs_df.shape, act_df.shape)      
      index_to_drop = [i for i in range(act_df.shape[0], obs_df.shape[0])]
      # print(index_to_drop)
      obs_df.drop(labels=index_to_drop, axis='index', inplace=True)
      observation_df = pd.concat([observation_df, obs_df], axis=0)
      action_df = pd.concat([action_df, act_df], axis=0)
    observation_df.fillna(0, inplace=True)
    observation_df.reset_index(drop=True, inplace=True)
    action_df.reset_index(drop=True, inplace=True)
    return observation_df, action_df  

# flat
print("Flat Terrain")
obs_files = sorted(glob.glob('logs/dataset/flat/[0-9]*_obs_df.csv'))
act_files = sorted(glob.glob('logs/dataset/flat/[0-9]*_action_df.csv'))

obs, act = merge_csv(obs_files, act_files)
#obs, act = merge_csv(['dataset/stairway/1_obs_df.csv', 'dataset/stairway/3_obs_df.csv'], ['dataset/stairway/1_action_df.csv', 'dataset/stairway/3_action_df.csv'])

obs.to_csv('logs/dataset/flat/obs_df.csv', index=False)
act.to_csv('logs/dataset/flat/action_df.csv', index=False)


# stairway
print("\nStairway Terrain")
obs_files = sorted(glob.glob('logs/dataset/stairway/[0-9]*_obs_df.csv'))
act_files = sorted(glob.glob('logs/dataset/stairway/[0-9]*_action_df.csv'))

obs, act = merge_csv(obs_files, act_files)
#obs, act = merge_csv(['dataset/stairway/1_obs_df.csv', 'dataset/stairway/3_obs_df.csv'], ['dataset/stairway/1_action_df.csv', 'dataset/stairway/3_action_df.csv'])

obs.to_csv('logs/dataset/stairway/obs_df.csv', index=False)
act.to_csv('logs/dataset/stairway/action_df.csv', index=False)


