import torch
import argparse
import numpy as np
import csv
import sys
sys.path.append("..") 
# from influence.data_collector import DataCollector
# from agents.random_agent import RandomAgent
# from simulators.warehouse.warehouse import Warehouse
import random
import matplotlib.pyplot as plt
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as sch

def init_weights(m):
    
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        # nn.init.uniform_(m.weight)
    elif isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)


class Network(nn.Module):
    """
    """
    def __init__(self, input_size, hidden_memory_size, n_sources, output_size, recurrent, seq_len, truncated):
        super().__init__()
        self.relu = nn.ReLU()
        self.recurrent = recurrent
        if self.recurrent:
            self.gru = nn.GRU(input_size, hidden_memory_size, batch_first=True)
        else:
            self.linear1 = nn.Linear(input_size*seq_len, hidden_memory_size)
        self.output_size = output_size
        if output_size > 1:
            self.softmax = nn.Softmax(dim=-1)
        else:
            self.softmax = nn.Sigmoid()
        self.hidden_memory_size = hidden_memory_size
        # self.linear2 = nn.Linear(hidden_memory_size, hidden_memory_size)
        self.linear3 = nn.Linear(hidden_memory_size, output_size*n_sources)
        self.n_sources = n_sources
        self.truncated = truncated
        self.reset()

    def forward(self, input_seq):
        input_seq = (input_seq - 0.5)/0.5
        if self.recurrent:
            out, self.hidden_cell = self.gru(input_seq, self.hidden_cell)
            if self.truncated:
                out = out[:, -1, :]
        else:
            out = self.relu(self.linear1(input_seq.flatten(start_dim=1)))
        # out = self.relu(self.linear2(out))
        logits = self.linear3(out).view(-1, self.n_sources, self.output_size)
        probs = self.softmax(logits).detach().numpy()
        return logits, probs
    
    def reset(self):
        self.hidden_cell = torch.zeros(1,1,self.hidden_memory_size)


class InfluenceNetwork(object):
    """
    """
    def __init__(self, parameters, data_path, agent_id, run_id):
        """
        """
        self._seq_len = parameters['seq_len']
        self._episode_length = parameters['episode_length']
        self._lr = parameters['lr']
        self._hidden_memory_size = parameters['hidden_memory_size']
        self._batch_size = parameters['batch_size']
        self.num_epochs = parameters['num_epochs']
        self.n_sources = parameters['n_sources']
        self.input_size = parameters['input_size']
        self.output_size = parameters['output_size']
        self.aug_obs = parameters['aug_obs']
        self.parameters = parameters
        self.inputs_file = data_path + 'inputs_' + str(agent_id) + '.csv'
        self.targets_file = data_path + 'targets_' + str(agent_id) + '.csv'
        self.recurrent = parameters['recurrent']
        self.truncated = self._seq_len < self._episode_length
        self.agent_id = agent_id
        self.model = Network(self.input_size, self._hidden_memory_size,
                             self.n_sources, self.output_size, self.recurrent,
                             self._seq_len, self.truncated)
        self.model.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
        # self.model.apply(init_weights)
        self.loss_function = nn.CrossEntropyLoss()
        if self.output_size == 1:
            self.loss_function = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        # self.scheduler = sch.StepLR(self.optimizer, step_size=100, gamma=0.1)
        # self.checkpoint_path = parameters['checkpoint_path'] + str(self.checkpoint_pathagent_id)

    def learn(self):

        # for layer in self.model.children():
        #     if hasattr(layer, 'reset_parameters'):
        #         layer.reset_parameters()
        inputs = self._read_data(self.inputs_file)
        targets = self._read_data(self.targets_file)
        input_seqs, target_seqs = self._form_sequences(inputs, targets)
        train_input_seqs, train_target_seqs, test_input_seqs, test_target_seqs = self._split_train_test(input_seqs, target_seqs)
        initial_loss, final_loss = self._train(train_input_seqs, train_target_seqs, test_input_seqs, test_target_seqs)
        self.trained = True
        # self._save_model()
        os.remove(self.inputs_file)
        os.remove(self.targets_file)
        return (initial_loss, final_loss)

    def test(self, inputs_file, targets_file):
        inputs = self._read_data(inputs_file)
        targets = self._read_data(targets_file)
        input_seqs, target_seqs = self._form_sequences(inputs, targets)
        loss = self._test(input_seqs, target_seqs)
        print(f'Test loss: {loss:10.8f}')
        os.remove(self.inputs_file)
        os.remove(self.targets_file)
        return loss

    
    def predict(self, obs):
        if self.recurrent:
            inputs = torch.cuda.FloatTensor(obs).view(1,1,-1)
        else:
            self.stack(obs)
            inputs = torch.cuda.FloatTensor(self.stacked_obs)
        _, probs = self.model(inputs)
        return probs[0]
    
    def reset(self):
        self.stacked_obs = np.zeros((1, self._seq_len, self.input_size))
        self.model.reset()
    
    def stack(self, obs):
        self.stacked_obs[:,:-1, :] = self.stacked_obs[:,1:,:]
        self.stacked_obs[0,-1, :] = obs
    
    def get_hidden_state(self):
        return self.model.hidden_cell.detach().numpy()


    ### Private methods ###        

    def _read_data(self, data_file):
        data = []
        with open(data_file) as data_file:
            csv_reader = csv.reader(data_file, delimiter=',')
            for row in csv_reader:
                data.append([int(element) for element in row])
        return data

    def _form_sequences(self, inputs, targets):
        n_episodes = len(inputs)//self._episode_length
        input_seq = []
        target_seq = []
        for episode in range(n_episodes):
            for seq in range(self._episode_length - (self._seq_len - 1)):
                start = episode*self._episode_length+seq
                end = episode*self._episode_length+seq+self._seq_len
                input_seq.append(inputs[start:end])
                if self.truncated:
                    target_seq.append(targets[end-1])
                else:
                    target_seq.append(targets[start:end])
        return input_seq, target_seq

    def _split_train_test(self, inputs, targets):
        test_size = int(0.1*len(inputs))
        train_inputs, train_targets = inputs[:-test_size], targets[:-test_size] 
        test_inputs, test_targets = inputs[-test_size:], targets[-test_size:]
        return train_inputs, train_targets, test_inputs, test_targets

    def _train(self, train_inputs, train_targets, test_inputs, test_targets):
        seqs = torch.cuda.FloatTensor(train_inputs)
        targets = torch.cuda.FloatTensor(train_targets)
        for e in range(self.num_epochs):
            permutation = torch.randperm(len(seqs))
            if e % 50 == 0:
                test_loss = self._test(test_inputs, test_targets)
                if e == 0:
                    initial_loss = test_loss
                print(f'epoch: {e:3} test loss: {test_loss:10.8f}')
                # print('lr: {1}'.format(e, self.optimizer.param_groups[0]['lr']))
            for i in range(0, len(seqs) - len(seqs) % self._batch_size, self._batch_size):
                indices = permutation[i:i+self._batch_size]
                seqs_batch = seqs[indices]
                targets_batch = targets[indices]
                self.model.hidden_cell = torch.zeros(1, self._batch_size, self._hidden_memory_size)
                seqs_batch.to(torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
                logits, probs = self.model(seqs_batch)
                if self.output_size > 1:
                    targets_batch = torch.argmax(targets_batch.view(-1, self.n_sources, self.output_size), dim=2).long().flatten()
                else:
                    targets_batch = targets_batch.view(-1, 1)
                logits = logits.flatten(end_dim=1)
                loss = self.loss_function(logits, targets_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # self.scheduler.step()
        final_loss = self._test(test_inputs, test_targets)
        print(f'epoch: {e+1:3} test loss: {final_loss:10.8f}')
        self.model.reset()
        return initial_loss, final_loss

    def _test(self, inputs, targets):
        inputs = torch.cuda.FloatTensor(inputs)
        targets = torch.cuda.FloatTensor(targets)
        loss = 0
        self.model.hidden_cell = torch.zeros(1, len(inputs), self._hidden_memory_size)
        logits, probs = self.model(inputs)
        if self.output_size > 1:
            targets = torch.argmax(targets.view(-1, self.n_sources, self.output_size), dim=2).long().flatten()
        else:
            targets = targets.view(-1, 1)
        logits = logits.flatten(end_dim=1)
        loss = self.loss_function(logits, targets)
        return loss.item()

    def _plot_prediction(self, prediction, target):
        prediction = prediction.detach().numpy()
        prediction = np.reshape(np.append(prediction, [prediction[5]]*19), (5,5))
        target = target.detach().numpy()
        target = np.reshape(np.append(target, [target[5]]*19), (5,5))
        if self.img1 is None:
            fig = plt.figure(figsize=(10,6))
            sub1 = fig.add_subplot(1, 2, 2)
            self.img1 = sub1.imshow(prediction, vmin=0, vmax=1)
            sub2 = fig.add_subplot(1, 2, 1)
            self.img2 = sub2.imshow(target, vmin=0, vmax=1)
            plt.tight_layout()
        else:
            self.img1.set_data(prediction)
            self.img2.set_data(target)
        plt.pause(0.5)
        plt.draw()

    def _save_model(self):
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        torch.save({'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, 
                    os.path.join(self.checkpoint_path, 'checkpoint'))
    
    def _load_model(self):
        checkpoint = torch.load(os.path.join(self.checkpoint_path, 'checkpoint'))
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        # torch.set_grad_enabled(False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']


if __name__ == '__main__':
    sys.path.append("..") 
    from agents.random_agent import RandomAgent
    from simulators.warehouse.warehouse import Warehouse
    from influence_dummy import InfluenceDummy
    from data_collector import DataCollector
    agent = RandomAgent(2)
    parameters = {'n_sources': 4, 'output_size': 1, 'aug_obs': False}
    parameters = read_parameters('./configs/influence.yaml')
    influence = InfluenceNetwork(parameters, './data/traffic/', None)
    # data_collector = DataCollector(agent, 'warehouse', 8, influence, './data/warehouse/', 0)
    # data_collector.run(parameters['dataset_size'], log=True)
    influence.train(2000)
