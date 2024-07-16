class LRLinearSchedule(object):

    def __init__(self, optimizer, total_steps, initial, final=0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.initial = initial
        self.final = final

    def update_learning_rate(self, step):

        learning_rate = (self.initial-self.final)*(1 - step/self.total_steps) + self.final
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate
            

class LinearSchedule(object):

    def __init__(self, total_steps, initial, final=0):
        self.total_steps = total_steps
        self.initial = initial
        self.final = final
    
    def update(self, step):     
        return (self.initial-self.final)*(1 - step/self.total_steps) + self.final
