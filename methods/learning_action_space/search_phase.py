import numpy as np
from .learning_phase import LearningPhase
from .utils import plotfig, distribution
import abc

class SearchPhase():
    # __metaclass__ = abc.ABCMeta
    def __init__(self, initial_sample=80, selects = 10, height_level=[400, 800, 1600, 3200], sample_func=None, get_net_acc=None, convert_func=None):
        self.sampleFunction = sample_func
        self.get_net_acc = get_net_acc
        self.convert_func = convert_func
        network = self.sampleFunction()
        network = network.reshape([1, -1])
        self.X = np.delete(network, 0, 0)
        self.y = np.array([])
        self.initial_sample = initial_sample
        self.selects = selects
        self.height_level = height_level
        self.current_select = 0
        self.net_acc = []
    # @abc.abstractmethod
    # def sampleFunction(self):
    #     return 
    # abstract method for inherit

    def selectSample(self):
        while len(self.y) < self.initial_sample:
            X_s = self.sampleFunction()
            print(X_s)
            yield X_s 

        self.classifier = LearningPhase(self.X, self.y, self.height(), 1)

        while True:
            self.current_select = 0 
            while self.current_select < self.selects:
                self.path_model, self.path_node = self.classifier.ucb_select()
                X_s = self.classifier.sample(self.path_model, self.path_node, self.sampleFunction)
                yield X_s 
    
            self.classifier = LearningPhase(self.X, self.y, self.height(), 1)
            print('--------------------current length------------------------')
            print('--------------------'+str(len(self.y))+'------------------------')
   
    def height(self):
        l = len(self.y)
        return np.searchsorted(self.height_level, l) + 1

    def back_propagate(self, network, acc):
        if acc != None:
            network = network.reshape([1, -1])
            acc = np.array([acc])
            self.X = np.concatenate([self.X, network])
            self.y = np.concatenate([self.y, acc])
        if len(self.y) > self.initial_sample:
            for n in self.path_model:
                n.n = n.n + 1
            self.current_select += 1

    def step_start_trigger(self):
        # define your method before learning action space
        pass

    def step_end_trigger(self):
        self.save_net_acc()

    def save_net_acc(self):
        pass
        # self.net_acc.append(['X':self.X, 'y':self.y])
        

        return
        if len(self.y)%100 == 0 and len(self.y) != 0 :
            with open(os.path.join(files,'trainXy.pkl'), 'wb+') as f:
                pickle.dump({'train': {'X':self.X, 'y':self.y}}, f) 

    def run(self, target_accuracy=1, max_samples=10000000):
        sample = self.selectSample()
        self.current_max_accuracy = 0 
        self.current_best_net = None
        while (self.current_max_accuracy < target_accuracy and len(self.y) < max_samples):
            self.step_start_trigger()
            network = next(sample) # Sampling a network for training
            if not isinstance(network, dict): 
                accuracy = self.get_net_acc(self.convert_func(network))             # Get the accuracy of the sampling network after training
            else:
                raise NotImplementedError
            self.back_propagate(network, accuracy)          # update the learning phase according to the network and it's accuracy
            if accuracy > self.current_max_accuracy:
                self.current_max_accuracy = accuracy
                self.current_best_net = network 
            self.step_end_trigger()

    def get_top_accuracy(self, k):
        top_k= []
        top_k_index = np.argsort(self.y)[::-1][:k]
        for index in top_k_index:
            top_k.append([self.X[index],self.y[index]])
        return top_k




