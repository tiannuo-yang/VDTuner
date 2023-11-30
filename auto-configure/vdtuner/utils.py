import sys 
sys.path.append("..") 

import joblib
from scipy.stats import qmc
import json
import numpy as np
import time
import subprocess as sp
import random
from configure import *

KNOB_PATH = r'/home/ytn/milvusTuning/auto-configure/whole_param.json'
RUN_ENGINE_PATH = r'/home/ytn/milvusTuning/vector-db-benchmark-master/run_engine.sh'

def LHS_sample(dimension, num_points, seed):
    sampler = qmc.LatinHypercube(d=dimension, seed=seed)
    latin_samples = sampler.random(n=num_points)

    return latin_samples

class KnobStand:
    def __init__(self, path) -> None:
        self.path = path
        with open(path, 'r') as f:
            self.knobs_detail = json.load(f)

    def scale_back(self, knob_name, zero_one_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            real_val = zero_one_val * (knob['max'] - knob['min']) + knob['min']
            return int(real_val), int(real_val)

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            enum_index = int(enum_size * zero_one_val)
            enum_index = min(enum_size - 1, enum_index)
            real_val = knob['enum_values'][enum_index]
            return enum_index, real_val
    
    def scale_forward(self, knob_name, real_val):
        knob = self.knobs_detail[knob_name]
        if knob['type'] == 'integer':
            zero_one_val = (real_val - knob['min']) / (knob['max'] - knob['min'])
            return zero_one_val

        elif knob['type'] == 'enum':
            enum_size = len(knob['enum_values'])
            zero_one_val = knob['enum_values'].index(real_val) / enum_size
            return zero_one_val

class StaticEnv:
    def __init__(self, model_path=['XGBoost_20knob_thro.model', 'XGBoost_20knob_prec.model'], knob_path=r'milvus_important_params.json') -> None:
        self.model_path = model_path
        self.get_surrogate(model_path)
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_surrogate(self, surrogate_path):
        # surrogate1, surrogate2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])
        self.model1, self.model2 = joblib.load(surrogate_path[0]), joblib.load(surrogate_path[1])

    def get_state(self, knob_vals_arr):
        Y1, Y2 = [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[0] for j,knob_val in enumerate(record)]
            print(f"Index parameters changed: {conf_value}")

            y1 = self.model1.predict([conf_value])[0]
            y2 = self.model2.predict([conf_value])[0]

            self.sampled_times += 1
            print(f'[{self.sampled_times}] {int(time.time()-self.t1)} {y1} {y2}')
            
            Y1.append(y1)
            Y2.append(y2)
        return np.concatenate((np.array(Y1).reshape(-1,1), np.array(Y2).reshape(-1,1)), axis=1)

class RealEnv:
    def __init__(self, bench_path=RUN_ENGINE_PATH, knob_path=KNOB_PATH) -> None:
        self.bench_path = bench_path
        self.knob_stand = KnobStand(knob_path)
        self.names = list(self.knob_stand.knobs_detail.keys())
        self.t1 = time.time()
        self.t2 = time.time()
        self.sampled_times = 0

        self.X_record = []
        self.Y1_record = []
        self.Y2_record = []
        self.Y_record = []

    def get_state(self, knob_vals_arr):
        Y1, Y2, Y3 = [], [], []
        for i,record in enumerate(knob_vals_arr):
            conf_value = [self.knob_stand.scale_back(self.names[j], knob_val)[1] for j,knob_val in enumerate(record)]
            
            index_value, system_value = conf_value[:9], conf_value[9:]
            index_name, system_name = self.names[:9], self.names[9:]

            index_conf = dict(zip(index_name,index_value))
            system_conf = dict(zip(system_name,system_value))

            configure_index(*filter_index_rule(index_conf))
            configure_system(filter_system_rule(system_conf))

            # print(f"Parameters changed to: {index_conf} {system_conf}")

            try:
                result = sp.run(f'sudo timeout 900 {RUN_ENGINE_PATH} "" "" glove-100-angular', shell=True, stdout=sp.PIPE)
                result = result.stdout.decode().split()
                y1, y2 = float(result[-2]), float(result[-3]) 
                # y1, y2 = 1698.4412378836437*(random.random()+0.5), 0.822103*(random.random()+0.5)
                
                self.Y1_record.append(y1)
                self.Y2_record.append(y2)
            except:
                y1, y2 = min(self.Y1_record), min(self.Y2_record)
            
            y3 = int(time.time()-self.t2)
            self.sampled_times += 1

            self.t2 = time.time()
            print(f'[{self.sampled_times}] {int(self.t2-self.t1)} {y1} {y2} {y3}')
            sp.run(f'echo [{self.sampled_times}] {int(self.t2-self.t1)} {index_conf} {system_conf} {y1} {y2} {y3} >> record.log', shell=True, stdout=sp.PIPE)

            Y1.append(y1)
            Y2.append(y2)
            Y3.append(y3)

        return np.array([Y1,Y2,Y3]).T

    def default_conf(self):
        return [self.knob_stand.scale_forward(k, v['default']) for k,v in self.knob_stand.knobs_detail.items()]

if __name__ == '__main__':
    print(type(LHS_sample(5,10)))

