from gym_minigrid.wrappers import *
from gym import wrappers

class FeatureVectorWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape 
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_shape[0]*obs_shape[1]*obs_shape[2],))

    def observation(self, obs):
        obs = obs[:,:,0].astype(int)
        obs[np.where(obs==1)] = 0
        obs[np.where(obs==2)] = 1
        obs[np.where(obs==5)] = 2
        obs[np.where(obs==6)] = -2
        obs = np.reshape(obs,-1)
        # dset = obs[np.where(obs == 5)]
        # dset = np.append(dset, obs[np.where(obs == 6)])
        # if len(dset) > 0:
        #     dset = np.mean(dset)
        # else:
        #     dset = 0
        # obs = np.append(obs, dset)
        return obs