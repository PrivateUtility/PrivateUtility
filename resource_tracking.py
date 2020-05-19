import os
import time
import pickle
import resource

class resource_tracking():
    def __init__(self, args):
        self.checkpoints = {}
        self.args = args

    def create_checkpoint(self, cp_name):
        # Collect information at current checkpoint
        curr_res = {}
        curr_res['proc_time'] = time.process_time()
        curr_res['proc_self_resource'] = resource.getrusage(resource.RUSAGE_SELF)
        curr_res['proc_child_resource'] = resource.getrusage(resource.RUSAGE_CHILDREN)
        
        # save checkpoint into class structure
        self.checkpoints[cp_name] = curr_res

    def save_log(self, save_path):
        sub_save_path = '{}/resources'.format(save_path)
        if not os.path.exists(sub_save_path):
            os.makedirs(sub_save_path)
        file_path = '{}/{}.p'.format(sub_save_path, self.args.run)
        
        pickle.dump(self.checkpoints, open(file_path, 'wb'))
