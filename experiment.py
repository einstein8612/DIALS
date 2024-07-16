import os
import sys

sys.path.append("..")
from influence.influence_network import InfluenceNetwork
from influence.influence_uniform import InfluenceUniform

# from simulators.vec_env import VecEnv
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecNormalize,
    VecFrameStack,
    DummyVecEnv,
)
from recurrent_policies.PPO import Agent, FNNPolicy, GRUPolicy, IAMGRUPolicy
import gym
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder
import numpy as np
import csv
import os
import time
from copy import deepcopy
from trainer import DistributedTraining, GlobalTraining, train_multi_agent
import psutil

# from sacred.settings import SETTINGS
# SETTINGS.CAPTURE_MODE = 'sys'


def log(dset, infs, data_path, learning_agent_ids):
    """
    Log influence dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    dset_array = np.swapaxes(dset, 0, 1)
    infs_array = np.swapaxes(infs, 0, 1)
    for i, agent_id in enumerate(learning_agent_ids):
        dset = dset_array[i]
        infs = infs_array[i]
        inputs_file = data_path + "inputs_" + str(agent_id) + ".csv"
        targets_file = data_path + "targets_" + str(agent_id) + ".csv"
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
    # MONGO_HOST = 'TUD-tm2'
    # MONGO_DB = 'distributed-simulation'
    # PKEY = '~/.ssh/id_rsa'
    # try:
    #     print("Trying to connect to mongoDB '{}'".format(MONGO_DB))
    #     server = SSHTunnelForwarder(
    #         MONGO_HOST,
    #         ssh_pkey=PKEY,
    #         remote_bind_address=('127.0.0.1', 27017)
    #         )
    #     server.start()
    #     DB_URI = 'mongodb://localhost:{}/distributed-simulation'.format(server.local_bind_port)
    #     # pymongo.MongoClient('127.0.0.1', server.local_bind_port)
    #     ex.observers.append(MongoObserver.create(DB_URI, db_name=MONGO_DB, ssl=False))
    #     print("Added MongoDB observer on {}.".format(MONGO_DB))
    # except pymongo.errors.ServerSelectionTimeoutError as e:
    # print(e)
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
                1,
            )
        elif self.parameters["policy"] == "IAMGRUPolicy":
            policy = IAMGRUPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                1,
                dset=self.parameters["dset"],
                dset_size=self.parameters["dset_size"],
            )
        elif self.parameters["policy"] == "GRUPolicy":
            policy = GRUPolicy(
                self.parameters["obs_size"],
                self.parameters["num_actions"],
                self.parameters["hidden_size"],
                self.parameters["hidden_size_2"],
                1,
            )

        self.agents = []
        for agent_id in self.parameters["learning_agent_ids"]:
            save_path = os.path.join(
                "saved_policies",
                str(self.parameters["env"]),
                str(self.parameters["simulator"]),
                str(agent_id),
            )
            self.agents.append(
                Agent(
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
                    save_path=save_path,
                    rollout_steps=self.parameters["rollout_steps"],
                )
            )

        global_sim_name = (
            self.parameters["env"] + ":global-" + self.parameters["name"] + "-v0"
        )
        self.global_simulator = gym.make(
            global_sim_name,
            seed=seed,
            learning_agent_ids=self.parameters["learning_agent_ids"],
        )

        if self.parameters["simulator"] == "distributed":

            self.data_path = os.path.join(
                parameters["influence"]["data_path"], str(_run._id)
            )
            self.dataset_size = parameters["influence"]["dataset_size"]

            self.local_simulators = []

            for i, agent_id in enumerate(self.parameters["learning_agent_ids"]):

                local_sim_name = (
                    self.parameters["env"] + ":local-" + self.parameters["name"] + "-v0"
                )

                if self.parameters["influence_model"] == "nn":
                    self.influence = []
                    influence = InfluenceNetwork(
                        parameters["influence"], self.data_path, agent_id, _run._id
                    )
                else:
                    influence = InfluenceUniform(parameters["influence"])

                self.local_simulators.append(
                    gym.make(
                        local_sim_name,
                        influence=influence,
                        seed=seed + i,
                        agent_id=agent_id,
                    )
                )

            self.trainer = DistributedTraining(self.agents, self.local_simulators)

        else:

            self.trainer = GlobalTraining(self.agents, self.global_simulator)

        arr = np.zeros((1024, 1024, 1024, 3), dtype=np.uint8)

    def run(self):

        total_steps = int(self.parameters["total_steps"])
        eval_freq = int(self.parameters["eval_freq"])

        train_steps = eval_freq
        if self.parameters["simulator"] == "distributed":
            influence_train_freq = int(self.parameters["influence_train_freq"])
            if eval_freq >= influence_train_freq:
                train_steps = influence_train_freq

        for step in range(0, total_steps + 1, train_steps):

            if (
                self.parameters["simulator"] == "distributed"
                and not self.parameters["untrained_influence"]
            ):
                if step % influence_train_freq == 0:
                    start = time.time()
                    self.collect_data(self.dataset_size, self.data_path)
                    initial_loss, final_loss = self.trainer.train_influence()
                    self._run.log_scalar("influence loss", initial_loss, step)
                    self._run.log_scalar("final loss", final_loss, step)
                    end = time.time()
                    print("Influence train time:", end - start)
            start = time.time()
            if step % eval_freq == 0:
                for agent in self.agents:
                    agent.save_policy()
                self.evaluate(step)

            end = time.time()
            print(f"[INF] Evaluate time for step {step}/{total_steps}:", end - start)
            start = time.time()
            self.agents = self.trainer.train(train_steps)
            end = time.time()
            print(f"[INF] Train time for step {step}/{total_steps}:", end - start)

        # self.trainer.close()

    def collect_data(self, dataset_size, data_path):
        """Collect data from global simulator"""
        print("[INF] Collecting data from global simulator...")
        n_steps = 0
        # copy agent to not alter hidden memory
        agents = deepcopy(self.agents)
        num_learning_agents = len(self.parameters["learning_agent_ids"])
        obs = self.global_simulator.reset(restart=True)
        while n_steps < dataset_size:
            done = [False] * num_learning_agents
            dset = []
            infs = []
            # NOTE: Episodes in all envs must terminate at the same time
            for agent in agents:
                agent.reset_hidden_memory([True])
            while not done[0]:
                n_steps += 1
                actions = []
                for i, agent in enumerate(agents):
                    action, _, _ = agent.choose_action(obs[i])
                    actions.append(action)
                obs, _, done, info = self.global_simulator.step(actions)
                dset.append(info["dset"])
                infs.append(info["infs"])
            log(dset, infs, data_path, self.parameters["learning_agent_ids"])
            obs = self.global_simulator.reset()
        print("[INF] Finished collecting data from global simulator.")

    def evaluate(self, step, collect_data=False):
        """Return mean sum of episodic rewards) for given model"""
        episode_rewards = []
        n_steps = 0
        # copy agent to not altere hidden memory
        agents = deepcopy(self.agents)
        num_learning_agents = len(self.parameters["learning_agent_ids"])
        print(f"[INF] Evaluating policy on global simulator for step {step}...")
        obs = self.global_simulator.reset(restart=True)
        while n_steps < self.parameters["eval_steps"]:
            reward_sum = np.array([0.0] * num_learning_agents)
            done = [False] * num_learning_agents
            # NOTE: Episodes in all envs must terminate at the same time
            for agent in agents:
                agent.reset_hidden_memory([True])
            while not done[0]:
                n_steps += 1
                actions = []
                for i, agent in enumerate(agents):
                    action, _, _ = agent.choose_action(obs[i])
                    actions.append(action)
                obs, reward, done, info = self.global_simulator.step(actions)
                if self.parameters["render"]:
                    self.global_simulator.render()
                    time.sleep(0.5)
                reward_sum += np.array(reward)
            obs = self.global_simulator.reset()
            episode_rewards.append(reward_sum)

        self._run.log_scalar("mean episodic return", np.mean(episode_rewards), step)
        print(f"[INF] Episode Rewards for step {step}:", np.mean(episode_rewards))

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


if __name__ == "__main__":
    ex = sacred.Experiment("distributed-simulation")
    ex.add_config("configs/default.yaml")
    add_mongodb_observer()

    @ex.automain
    def main(parameters, seed, _run):
        exp = Experiment(parameters, _run, seed)
        exp.run()
