import time
import numpy as np
import multiprocessing as mp
import multiprocessing.connection
from multiprocessing import Process
from multiprocessing.pool import ThreadPool as Pool

def train_single_agent(agent_id, agent_dict, agent, env, training_steps=1.0e+5):
    
    obs = env.reset(restart=True)
    step = 0
    done = [True]

    while step <= training_steps:
        
        rollout_step = 0
        while rollout_step < agent.rollout_steps:

            if agent.policy.recurrent:
                agent.reset_hidden_memory(done)
                hidden_memory = agent.policy.hidden_memory
            else:
                hidden_memory = None
            action, value, log_prob = agent.choose_action(obs)
            new_obs, reward, done, _ = env.step(action)
            agent.add_to_memory(obs, action, reward, done, value, log_prob, hidden_memory)
            obs = new_obs
            rollout_step += 1
            step += 1

            if done[0]:
                obs = env.reset()
        agent.bootstrap(obs)

        if agent.buffer.is_full:
            agent.update()
    agent_dict[agent_id] = agent
    

def train_multi_agent(agents, env, training_steps=1.0e+3):
  
    obs = env.reset(restart=True)
    step = 0
    done = [True]*len(agents)
    while step <= training_steps:
        
        rollout_step = 0
        while rollout_step < agents[0].rollout_steps:
            
            hidden_memories = []
            actions = []
            values = []
            log_probs = []
            for i, agent in enumerate(agents):
                if agent.policy.recurrent:
                    agent.reset_hidden_memory([done[i]])
                    hidden_memories.append(agent.policy.hidden_memory)
                else:
                    hidden_memories.append(None)
                action, value, log_prob = agent.choose_action(obs[i])
                actions.append(action)
                values.append(value)
                log_probs.append(log_prob)

            new_obs, reward, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.add_to_memory(obs[i], actions[i], [reward[i]], [done[i]], values[i], log_probs[i], hidden_memories[i])
            
            obs = new_obs
            rollout_step += 1
            step += 1

            if done[0]:
                obs = env.reset()

        for agent in agents:
            agent.bootstrap(obs[i])
            if agent.buffer.is_full:
                agent.update()

    return agents


class DistributedTraining(object):
    
    def __init__(self, agents, sims):
        # self.local_trainers = [LocalTrainer(agent, env) for agent, env, in zip(agents, envs)]
        # self.pool =  Pool(processes=len(agents))
        self.agents = agents
        self.sims = sims
            
    def train(self, training_steps):

        # for local_trainer in self.local_trainers:
        #     local_trainer.child.send(('train_agent', training_steps))
        #     agents.append(local_trainer.child.recv())
        # pool = Pool()#processes=len(self.agents))
        # outputs = []
        # for i in range(len(self.agents)):
        #     outputs.append(pool.apply_async(train_single_agent, (self.agents[i], self.envs[i], training_steps)))

        # agents = [output.get() for output in outputs]
        # pool.close()
        manager = multiprocessing.Manager()
        agent_dict = manager.dict()
        processes = []
        for i in range(len(self.agents)):
            p = Process(target=train_single_agent, args=(i, agent_dict, self.agents[i], self.sims[i], training_steps))
            processes.append(p)
            p.start()
            # p.join()
        #     processes.append(p)
        #     p.start()
        for p in processes:
            p.join()
            # p.close()
        # with Pool() as pool:
            # agents = pool.starmap(train_single_agent, zip(agents, envs))
        # with Pool() as pool:
            # outputs = []
        #     for agent, env in zip(agents, envs):
        #         outputs.append(pool.apply_async(train_single_agent, (agent, env)))
        #     agents = [output.get() for output in outputs]
        # self.agents = agents
        agent_dict = dict(sorted(agent_dict.items())) # agents may be returned in the wrong order
        self.agents = list(agent_dict.values())
        return self.agents

    def train_influence(self):
        outputs = []

        with Pool() as pool:
            for sim in self.sims:
                outputs.append(pool.apply_async(sim.influence.learn, ()))
            pool.close()
            pool.join()
        initial_loss_mean = 0
        final_loss_mean = 0
        for output in outputs:
            initial_loss, final_loss = output.get()
            initial_loss_mean += initial_loss/len(outputs)
            final_loss_mean += final_loss/len(outputs)
        # for i in range(len(self.sims)):
            # self.sims[i].influence.model = influence_models[i]
        # for env in self.sims:
        #     loss = env.influence.learn()
        
        # 
        # for i in range(len(self.sims)):
        #     self.sims[i].load_influence_model()

        return initial_loss_mean, final_loss_mean

    def close(self):

        for sim in self.sims:
            sim.close()
    

class GlobalTraining(object):

    def __init__(self, agents, sim):

        self.agents = agents
        self.sim = sim
    
    def train(self, training_steps):

        self.agents = train_multi_agent(self.agents, self.sim, training_steps)
        
        return self.agents

    def close(self):

        self.sim.close()
