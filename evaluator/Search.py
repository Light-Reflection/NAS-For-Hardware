from dmmo.methods import SearchPhase
# from dmmo.supernet import 
import numpy as np
# from utils import load_model
from dmmo.evaluator.utils import create_exp_dir
import pickle
from dmmo.generator.utils import produce_channel_encoding, produce_resolution_encoding, produce_op_encoding, produce_ksize_encoding
# from dmmo.generator.Supernet import supernet
# from dmmo.generator.automodel import 

'''
    Example:
        modify infer.py Line 38 ==> parser.add_argument('--config', default='./models/supernet/example_config/infer_cha_supernet.yaml', help='config dir')
        modify ./models/supernet/example_config/infer_cha_supernet.yaml model_kw ==> model_kw: {'num_classes': 10, 'layers': 12, 'division': 8, 'width_factor': 2,'status': 'infer', 'sample_method': 'None'} 
        python example_action_space --load ./best_weights.pt 
'''

params_threshold = 3.1 # M 
mac_threshold = 0.043  # GMac

class Searcher(object):
    """docstring for Searcher"""
    def __init__(self, method, initial_sample=80, selects=19, height_level=[400, 800, 1600, 3200], sample_func=None, get_net_acc=None, convert_func=None):
        super(Searcher, self).__init__()
        if method == 'AS': # AS: Action Spcae
            # params: initial_sample, selects, height_level
            self._method = SearchPhase(initial_sample=initial_sample, selects=selects, height_level=height_level, sample_func=sample_func, get_net_acc=get_net_acc, convert_func=convert_func)
        else:
            raise NotImplementedError
    def run(self, max_samples,target_acc):
        self._method.run(target_acc, max_samples)    

    def get_top_accuracy(self, k):
        return self._method.get_top_accuracy(k)

# def get_net_acc(net_id):
#     model = 
#     acc = model.top1.avg
#     return acc


# # ===== define searcher 
# class ActionSpace(SearchPhase):
#     def __init__(self, initial_sample=80, selects = 10, height_level=[400, 800, 1600, 3200], sample_func=None, get_net_acc=None, convert_func=None): 
#         super(ActionSpace, self).__init__(initial_sample, selects, height_level, sample_func, get_net_acc, convert_func) 

#     # def sampleFunction(self, sample_func):
#     #     return np.array(convert_dict_to_list(self._net_dict))

#     # def train_test(self, net_dict):
#     #     # get your net encoding performance
#     #     # net_dict is
#     #     return self._get_net_acc(self._net_dict)

#     # def step_start_trigger(self):
#     #     model.logger.info('current sample net nums:%d and best_acc: %.2f',len(self.y), self.current_max_accuracy) 
#     # The model not have logger

#     def step_end_trigger(self):
#         # TODO: add logger
#         # if self.y[-1] == self.current_max_accuracy:
#         #     model.logger.info('******* Update max accuracy *******')
#         if len(self.y)%100 == 0 and len(self.y) != 0 :
#             with open(os.path.join(files,'trainXy.pkl'), 'wb+') as f:
#                 pickle.dump({'train': {'X':self.X, 'y':self.y}}, f) 


class EASearch():
    def __init__(self):
        pass

