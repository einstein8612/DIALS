from logging import INFO
from PPO.agent import Agent
from PPO.other_policies import IAMGRUPolicy_separate
import numpy as np
import os
import yaml
import os
import sys

sys.path.append("..")
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack
from PPO import (
    Agent,
    FNNPolicy,
    GRUPolicy,
    IAMGRUPolicy_dynamic,
    IAMGRUPolicy,
    LSTMPolicy,
    IAMLSTMPolicy,
    IAMGRUPolicy_separate,
)
import gym
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder
import numpy as np
import csv
import os
from copy import deepcopy
import time
from gym import spaces

# from gym_minigrid.wrappers import *
from gym import wrappers

# from environments.minigrid.wrappers import FeatureVectorWrapper


def generate_path(path):
    """
    Generate a path to store e.g. logs, models and plots. Check if
    all needed subpaths exist, and if not, create them.
    """
    result_path = os.path.join("../results", path)
    model_path = os.path.join("../models", path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    return path


def log(dset, infs, data_path):
    """
    Log influence dataset
    """
    generate_path(data_path)
    inputs_file = data_path + "inputs.csv"
    targets_file = data_path + "targets.csv"
    dset = np.reshape(np.swapaxes(dset, 0, 1), (-1, np.shape(dset)[2]))
    infs = np.reshape(np.swapaxes(infs, 0, 1), (-1, np.shape(infs)[2]))
    with open(inputs_file, "a") as file:
        writer = csv.writer(file)
        for element in dset:
            writer.writerow(element)
    with open(targets_file, "a") as file:
        writer = csv.writer(file)
        for element in infs:
            writer.writerow(element)


def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    MONGO_HOST = "TUD-tm2"
    MONGO_DB = "scalable-simulations"
    PKEY = "~/.ssh/id_rsa"
    try:
        print("Trying to connect to mongoDB '{}'".format(MONGO_DB))
        server = SSHTunnelForwarder(
            MONGO_HOST, ssh_pkey=PKEY, remote_bind_address=("127.0.0.1", 27017)
        )
        server.start()
        DB_URI = "mongodb://localhost:{}/scalable-simulations".format(
            server.local_bind_port
        )
        # pymongo.MongoClient('127.0.0.1', server.local_bind_port)
        ex.observers.append(MongoObserver.create(DB_URI, db_name=MONGO_DB, ssl=False))
        print("Added MongoDB observer on {}.".format(MONGO_DB))
    except pymongo.errors.ServerSelectionTimeoutError as e:
        print(e)
        print("ONLY FILE STORAGE OBSERVER ADDED")
        from sacred.observers import FileStorageObserver

        ex.observers.append(FileStorageObserver.create("saved_runs"))


class Experiment(object):
    """
    Creates experiment object to interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters, _run, seed):
        """ """
        self._run = _run
        self._seed = seed
        self.parameters = parameters["main"]

        if self.parameters["policy"] == "FNNPolicy":
            policy = FNNPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["num_workers"],
            )
        elif self.parameters["policy"] == "IAMGRUPolicy_dynamic":
            policy = IAMGRUPolicy_dynamic(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["hidden_memory_size"],
                self.parameters["attention_size"],
                self.parameters["temperature"],
                self.parameters["num_workers"],
                dset=self.parameters["dset"],
                dset_size=self.parameters["dset_size"],
            )
        elif self.parameters["policy"] == "IAMGRUPolicy":
            policy = IAMGRUPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["hidden_memory_size"],
                self.parameters["num_workers"],
                dset=self.parameters["dset"],
                dset_size=self.parameters["dset_size"],
            )

        elif self.parameters["policy"] == "IAMLSTMPolicy":
            policy = IAMLSTMPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["hidden_memory_size"],
                self.parameters["num_workers"],
                dset=self.parameters["dset"],
                dset_size=self.parameters["dset_size"],
            )

        elif self.parameters["policy"] == "GRUPolicy":
            policy = GRUPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["num_workers"],
            )

        elif self.parameters["policy"] == "LSTMPolicy":
            policy = LSTMPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                self.parameters["num_workers"],
            )

        self.agent = Agent(
            policy=policy,
            memory_size=self.parameters["memory_size"],
            batch_size=self.parameters["batch_size"],
            seq_len=self.parameters["seq_len"],
            num_epoch=self.parameters["num_epoch"],
            learning_rate=self.parameters["learning_rate"],
            total_steps=self.parameters["total_steps"],
            clip_range=self.parameters["epsilon"],
            entropy_coef=self.parameters["beta"],
            load=self.parameters["load_policy"],
            rollout_steps=self.parameters["rollout_steps"],
        )

        self.seed = seed
        np.random.seed(seed)
        self.env = self.create_env()

    def create_env(self):
        env_id = self.parameters["env"] + ":" + self.parameters["name"] + "-v0"
        env = SubprocVecEnv(
            [
                self.make_env(self.parameters["env"], env_id, i, self.seed)
                for i in range(self.parameters["num_workers"])
            ],
            "spawn",
        )
        env = VecNormalize(env, norm_reward=False, norm_obs=False)
        if self.parameters["framestack"]:
            env = VecFrameStack(env, n_stack=self.parameters["n_stack"])

        return env

    def make_env(self, env_name, env_id, rank, seed=0, influence=None):
        """
        Utility function for multiprocessed env.
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environments you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """

        def _init():
            if env_name == "minigrid":
                env = gym.make(id="minigrid:MiniGrid-MemoryS11-modified-v0")
                env = ImgObsWrapper(env)  # Get rid of the 'mission' field
                env = FeatureVectorWrapper(env)
                env.seed(seed + np.random.randint(1.0e6))
            else:
                env = gym.make(env_id, seed=seed + np.random.randint(1.0e6))
                # env = Monitor(env, './logs')
                env.seed(seed + rank)
            return env

        # set_global_seeds(seed)
        return _init

    def learn(self):

        obs = self.env.reset()
        step = 0
        episode_reward = 0
        episode_step = 0
        episode = 1
        done = [False] * self.parameters["num_workers"]

        while step <= self.parameters["total_steps"]:

            rollout_step = 0

            while rollout_step < self.agent.rollout_steps:

                if step % self.parameters["eval_freq"] == 0:
                    self.evaluate(step)
                    self.agent.save_policy()
                #    self.env.reset()

                if self.agent.policy.recurrent:
                    self.agent.reset_hidden_memory(done)
                    hidden_memory = self.agent.policy.hidden_memory
                else:
                    hidden_memory = None

                action, value, log_prob = self.agent.choose_action(obs)
                new_obs, reward, done, _ = self.env.step(action)

                self.agent.add_to_memory(
                    obs, action, reward, done, value, log_prob, hidden_memory
                )
                obs = new_obs
                rollout_step += 1
                step += 1
                episode_step += 1
                episode_reward += self.env.get_original_reward()[0]

                if done[0]:
                    self.print_results(episode_reward, episode_step, step, episode)
                    episode_reward = 0
                    episode_step = 0
                    episode += 1

            self.agent.bootstrap(obs)

            if self.agent.buffer.is_full:
                self.agent.update()

    def print_results(self, episode_return, episode_step, global_step, episode):
        """
        Prints results to the screen.
        """
        print(
            ("Train step {} of {}".format(global_step, self.parameters["total_steps"]))
        )
        print(("-" * 30))
        print(("Episode {} ended after {} steps.".format(episode, episode_step)))
        print(("- Total reward: {}".format(episode_return)))
        print(("-" * 30))

    def evaluate(self, step):
        """Return mean sum of episodic rewards) for given model"""
        episode_rewards = []
        n_steps = 0
        # copy agent to not altere hidden memory
        agent = deepcopy(self.agent)
        # eval_env = self.create_env()
        eval_env = self.env
        print("Evaluating policy...")
        obs = eval_env.reset()
        done = [True] * self.parameters["num_workers"]
        reward_sum = np.array([0.0] * self.parameters["num_workers"])
        while n_steps < self.parameters["eval_steps"] // self.parameters["num_workers"]:
            # NOTE: Episodes in all envs must terminate at the same time
            if agent.policy.recurrent:
                agent.reset_hidden_memory(done)
            n_steps += 1
            action, _, _ = agent.choose_action(obs)
            obs, _, done, info = eval_env.step(action)
            if self.parameters["render"]:
                eval_env.render()
                time.sleep(0.5)
            # print(obs)
            # breakpoint()
            reward_sum += eval_env.get_original_reward()
            for i, d in enumerate(done):
                if d:
                    episode_rewards.append(reward_sum[i])
                    reward_sum[i] = 0
        print(episode_rewards)
        self._run.log_scalar("mean episodic return", np.mean(episode_rewards), step)
        # eval_env.close()
        print("Done!")


if __name__ == "__main__":
    ex = sacred.Experiment("scalable-simulations")
    ex.add_config("configs/default.yaml")
    add_mongodb_observer()

    @ex.automain
    def main(parameters, seed, _run):
        exp = Experiment(parameters, _run, seed)
        exp.learn()
