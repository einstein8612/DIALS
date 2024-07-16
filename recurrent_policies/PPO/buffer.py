import numpy as np
import random

class Buffer(dict):
    

    def __init__(self, memory_size):
        
        self.memory_size = memory_size
        

    def __getitem__(self, key):
        if key not in self.keys():
            self[key] = list()
        return super(Buffer, self).__getitem__(key)

    @property
    def is_full(self):
        """
        Check whether buffer is full.
        """
        if 'returns' not in self.keys():
            return False
        else:
            return len(self['returns']) >= self.memory_size

    def empty(self):
        for key in self.keys():
            self[key] = []

    def sample(self, b, batch_size, seq_len, shuffled_buffer, keys=None):
        """
        """
        batch = {}
        n_sequences = batch_size // seq_len
        if keys is None:
            keys = self.keys()
        for key in keys:
            batch[key] = []
            for seq in range(b*n_sequences, (b+1)*n_sequences):
                # start = s*seq_len + b*batch_size
                # end = (s+1)*seq_len + b*batch_size
                batch[key].append(shuffled_buffer[key][seq])
            # permut dimensions workers-seq to mantain sequence order
            # axis = np.arange(np.array(batch[key]).ndim)
            # axis[1], axis[2] = axis[2], axis[1]
            # try:
            batch[key] = np.swapaxes(batch[key], 1, 2)
            # except:
            #     print(key)
            #     print(np.shape(batch[key]))
            #     print(seq)
            #     print(b)
            # batch[key] = np.transpose(batch[key], axis)
        return batch

    # def shuffle(self, seq_len):
    #     """
    #     """
    #     n = len(self['returns'])
    #     # Only include complete sequences
    #     indices = np.arange(0, n - n % seq_len, seq_len)
    #     workers = np.shape(self['returns'])[1]
    #     shuffled_buffer = dict()
    #     for w in range(workers):
    #         random.shuffle(indices)
    #         for key in self.keys():
    #             shuffled_list = list()
    #             if key not in shuffled_buffer.keys():
    #                 shuffled_buffer[key] = list()
    #             for i in indices:
    #                 shuffled_list.extend(np.array(self[key])[i:i+seq_len, w])
    #             shuffled_buffer[key].append(shuffled_list)
    #     for key in self.keys():
    #         self[key] = np.swapaxes(shuffled_buffer[key], 0, 1)
    
    def shuffle(self, seq_len):
        """
        """
        n = len(self['returns'])
        # Only include complete sequences
        indices = np.arange(0, n - n % seq_len, seq_len)
        random.shuffle(indices)
        shuffled_buffer = {}
        for key in self.keys():
            shuffled_buffer[key] = []
            for i in indices:
                shuffled_buffer[key].append(self[key][i:i+seq_len])
        return shuffled_buffer

    def get_last_entries(self, t, keys=None):
        """
        """
        if keys is None:
            keys = self.keys()
        batch = {}
        for key in keys:
            batch[key] = self[key][-t:]
        return batch