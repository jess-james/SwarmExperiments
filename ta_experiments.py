from datetime import date
import datetime as dt
import cftime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import xarray as xr
import itertools
#from geopy.geocoders import Nominatim
import random
from collections import defaultdict, Counter
from heapq import heappush,heappop
import re
import os

#Import model anomaly data
models = {}
model_names = ["bccrbcm20", "ccmacgcm31t47", "csiromk35", "gissaom","mpiecham5", "mricgcm232a", "ncarccsm30", "cnrmcm3run1","gissmodelerrun4","ncarccsm30run9", "inmcm30", "cnrmcm3", "iapfgoals10g", "gissmodeler", "miroc32medres"]
for model_name in model_names:
    models[model_name] = pd.read_pickle("data/"+model_name+"_anom_month.pkl")

nasa = pd.read_pickle("data/nasagistemp.pkl")
directory = ''


#Loss Functions
def L(y, y_hat):
    return np.sqrt(np.square((y-y_hat)))

def Llog(y, y_hat):
    if type(y_hat) is list:
        c = np.zeros(len(y_hat))
        for i in range(0,len(y_hat)):
            c[i] = -y*np.log(y_hat[i]) - (1-y)*np.log(1-y_hat[i])
        return c
    return -y*np.log(y_hat) - (1-y)*np.log(1-y_hat)

def L01(y, y_hat):
    C = (np.round(y_hat)==y)
    return C.astype(np.int)

#Learners
class static_expert_learner:
    def __init__(self, n_experts, eta, c, tasks):
        self.n = n_experts
        self.eta = eta
        self.c = c
        self.W = [np.ones(n_experts)/n_experts for i in range(tasks)]
        self.s = tasks

    def update(self, x, y, task):
        self.W[task] *= np.exp(-self.eta*L(y,x))

    def predict(self, x, task):
        #Predict using Vovk's prediction
        '''v = self.W[task]/np.sum(self.W[task])
        d0 = -self.c*np.log(np.exp(-self.eta*L(0,x)) @ v)
        d1 = -self.c*np.log(np.exp(-self.eta*L(1,x)) @ v)
        return 0.5*(np.sqrt(d0) + 1 - np.sqrt(d1))'''
        v = self.W[task]/np.sum(self.W[task])
        return v @ x

class fixed_share_learner(static_expert_learner):
    def __init__(self, n_experts, eta, c, alpha, tasks):
        super().__init__(n_experts, eta, c, tasks)
        self.alpha = alpha

    def update(self, x, y, task):
        super().update(x, y, task)
        pool = self.alpha*np.sum(self.W[task])
        self.W[task] = (1 - self.alpha)*self.W[task] + (1/(self.n-1))*(pool - self.alpha*self.W[task])

class variable_share_learner(static_expert_learner):
    def __init__(self, n_experts, eta, c, alpha, tasks):
        super().__init__(n_experts, eta, c, tasks)
        self.alpha = alpha

    def update(self, x, y, task):
        super().update(x, y, task)
        pool = (1 - np.power(1 - self.alpha, L(y, x))) @ self.W[task]
        self.W[task] = np.power(1 - self.alpha, L(y, x))*self.W[task]  + (1/(num_experts-1))*(pool - (1 - np.power(1 - self.alpha, L(y, x)))*self.W[task])

class swarm():
    def __init__(self, s, n, m, k, T, eta=2):
        self.n = n
        self.s = s
        self.k = k
        self.pi = np.ones(n)/n
        self.mu = 1/m
        self.W = self.mu*np.ones((s,n))/m
        self.theta = 1 - (k/(T-s))
        self.phi = k/((m-1)*(T-s)) if m > 1 else k/((1)*(T-s))
        #self.eta = np.sqrt((m*np.log(n/m)+s*m*n*(1/m)+(T-s)*n*(k/(T-s))+(m-1)*(T-s)*n*(k/((m-1)*(T-s))))*(2/Tm))
        self.eta = eta

    def predict(self, x, task):
        v = (self.pi*self.W[task])/(self.pi @ self.W[task])
        return v @ x

    def update(self, x, y, task):
        c = L(y, x)
        delta = self.W[task]*np.exp(-self.eta*c)
        beta = (self.pi@self.W[task])/(self.pi@delta)
        epsilon = 1 - self.W[task] + beta*delta
        self.pi = self.pi*epsilon
        self.W[task] = (self.phi*(1-self.W[task]) + self.theta*beta*delta)*(1/epsilon)
        self.W = self.W/np.sum(self.W)

def rearrange(m,k,s):
    mode_list = [i%m for i in range(int((k+1)* s))]
    count = Counter(mode_list)
    priority_queue = []
    for mode, index in count.items():
        heappush(priority_queue, (-index, mode))
    mixed_list= []
    last = (0, '-')
    ind = 0
    while priority_queue:
        if ind >= len(mode_list):
          break
        most_mode = heappop(priority_queue)
        mixed_list.append(most_mode[1])
        if last[0] < 0:
            heappush(priority_queue, last)
        last = (most_mode[0]+1, most_mode[1])
        ind += 1
    if ind != len(mode_list):
        return None
    elif isinstance(mode_list, list):
        trials_per_task = int((k / s) + 1)
        task_modes = [[mixed_list.pop() for j in range(trials_per_task)] for i in range(s)]
        for task in task_modes:
            random.shuffle(task)
        return task_modes


#Generate experiment data based upon given models
class experiment:
    def __init__(self, learners, learner_names, experiment_params, cache=True):
        self.learners = learners
        self.learner_names = learner_names
        self.s, self.n, self.m, self.k, self.T = experiment_params
        self.local_t = np.zeros(self.s, np.int64)
        self.local_k = np.zeros(self.s, np.int64)
        self.experiment_type = experiment_type
        self.cache = cache

    def environment_data(self):
        print("Compiling environment data...")
        
        #Import data if stored
        if True and os.path.isfile("/Users/jessc/yr4/dissertation/data/envdataX3.npy") and os.path.isfile("/Users/jessc/yr4/dissertation/data/envdataY3.npy"):
                with open("/Users/jessc/yr4/dissertation/data/envdataX.npy", 'rb') as f:
                    X = np.load(f, allow_pickle=True)
                with open("/Users/jessc/yr4/dissertation/data/envdataY.npy", 'rb') as f:
                    Y = np.load(f, allow_pickle=True)
                    
        #Else generate data
        else:
            local_t = np.zeros(self.s, np.int64)
            local_mode = np.zeros(self.s, np.int64)
            
            Y = []
            X = []
            area_size = 10
        
        #Generate locations
        locs = []
        for location in range(self.s):
	        new_location = [np.random.randint(-90,90),np.random.randint(-180,180) ]
	        for existing_location in existing_locations:
	            if math.sqrt( ((new_location[0]-locs[0])**2)+((new_location[1]-locs[1])**2) ) < area_size:
	                break
	        else:
	            locs.append(new_location)
              
          #Import data for each location
          for task, (lat, lon) in enumerate(loxs):
              tasky = nasa[(nasa['time'].dt.year > 1989) & (nasa['time'].dt.year < 2000) & (nasa['latbin']==lat) & (nasa['lonbin']==lon)].groupby(pd.Grouper(key='time', freq="M"))['tempanomaly'].mean().array
              taskx=[]
              for model in models.values():
                  include = [(model['latbin']>lat-area_size) & (model['latbin']<lat+area_size) & (model['lonbin']>(lon-area_size)) & (model['lonbin']<(lon+area_size))]
                  include = include.groupby(pd.Grouper(key='time', freq="M"))['ta_anom'].mean().array#groupby(nasa['time'].dt.year, axis=1)['ta'].transform('mean')
                  taskx.append(include)
              for x in range(120):
                  year_vals = [taskx[i][x] for i in range(self.n)]#,taskx[1][x],taskx[2][x]]
                  X.append(year_vals)
              Y = np.concatenate((Y, tasky))
          newX = [X[(i%tasks)*120 + (i//tasks)] for i in range(120*tasks)]
          newY = [Y[(i%tasks)*120 + (i//tasks)] for i in range(120*tasks)]
          X = newX
          Y = newY
          np.save("data/envdataX.npy", X)
          np.save("data/envdataY.npy", Y)
      return X,Y

    #Return the task from the task vector and its local time for the current global trial
    def step(self,tau):
            task = (tau%tasks)
            t = self.local_t[task]
            self.local_t[task] += 1
            return task, t
          
         
for eta in [0.5, 1, 2, 5, 10]:
    for switch in [10, 20, 50, 100, 200, 500]:
        #Experiment Parameters
        locations = 10
        switches=[locations*2, locations*10, locations*20, locations*120]
        #Experiment set up
        global_trials = 120*10
        num_experts = len(models)
        eta = eta
        c = 1/eta
        global_switches = switch
        tasks = 10
        num_trials = global_trials/tasks
        modes = num_experts
        expert_list = np.arange(modes)
        
        #Initialise learners
        fixed_alpha = global_switches/(num_trials-1)
        expected_loss = 20
        expected_switches = global_switches
        var_alpha = expected_switches/(2*expected_switches + expected_loss)
        swarm_learner = swarm(tasks, num_experts, modes, global_switches, global_trials)
        this_variable_share_learner = variable_share_learner(num_experts, eta, c, var_alpha, tasks)
        this_fixed_share_learner = fixed_share_learner(num_experts, eta, c, fixed_alpha, tasks)
        learners = [swarm_learner, this_variable_share_learner, this_fixed_share_learner]
        learner_names = ['Swarm', 'Variable Share Learner', 'Fixed Share Learner', 'Average Expert', 'Best Partition', 'Best Expert']
        
        
        #Generate data
        basic_experiment = experiment(learners, learner_names, (tasks,num_experts,modes,global_switches,global_trials))
        X, Y = basic_experiment.environment_data()
        
        #Initialise storage for learner loss and weights 
        weights = {}
        losses = {}
        ground_val = pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials))
        for learner_name in learner_names:
            weights[learner_name] = {}
            losses[learner_name] = pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials))
            for i in range(tasks):
                weights[learner_name][i] = pd.DataFrame(0.0, columns = np.arange(num_experts), index = np.arange(num_trials))
        for model_name in model_names:
            prediction[model_name] = pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials))
            losses[model_name] = pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials))
        learner_loss = np.zeros(len(learners), np.float64)
        task_loss = [[] for i in range(tasks)]
        predictions = {name: pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials)) for name in learner_names}
        predictions = {name: pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials)) for name in model_names}
        
        #Run experiments
        for tau in range(global_trials):
            task, t = basic_experiment.step(tau)
            if (task<11):
                ground_val.loc[t, task] = Y[tau]
                #Calculate learner loss
                for key, learner in enumerate(learners):
                    y_hat = learner.predict(X[tau], task)
                    predictions[learner_names[ker]].loc[t, task] = y_hat
                    learner_loss[key] += np.sum(L(Y[tau], y_hat))
                    learner.update(X[tau],Y[tau],task)
                    losses[learner_names[key]].loc[t, task] = losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                    weights[learner_names[key]][task].loc[t,:] = learner.W[task]/np.sum(learner.W[task])

                #Calculate model loss and expert data
                if t>0:
                    for ind, name in enumerate(model_names):
                        prediction[name].loc[t,task] = X[tau][ind]
                        losses[name].loc[t,task] = losses[name].loc[t-1, task] + L(X[tau][ind], y_hat)
                    losses['Average Expert'].loc[t, task] = np.average([losses[name].loc[t, task] for name in model_names]) #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                    losses['Best Partition'].loc[t, task] = losses['Best Partition'].loc[t-1, task] + np.min([L(X[tau][i],Y[tau]) for i in range(num_experts)])
                else:
                    for ind, name in enumerate(model_names):
                        x=X[tau][ind]
                        prediction[name].loc[t,task] = x
                        losses[name].loc[t,task] = L(X[tau][ind], y_hat)
                    losses['Average Expert'].loc[t, task] = np.average([losses[name].loc[t, task] for name in model_names]) #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                    losses['Best Partition'].loc[t, task] = np.min([L(X[tau][i],Y[tau]) for i in range(num_experts)])
      
        w_e_by_task = []
        b_e_by_task = []
        for task in range(tasks): 
            print("Task number: {}".format(task))
            worst_expert_loss = 0
            best_expert_loss = 999999
            w_e = None
            b_e = None
            print(task)
            for model_name in model_names:
                if losses[model_name].loc[num_trials-1, task] > worst_expert_loss:
                    worst_expert_loss = losses[model_name].loc[num_trials-1, task]
                    w_e = losses[model_name]
                if losses[model_name].loc[num_trials-1, task] < best_expert_loss:
                    best_expert_loss = losses[model_name].loc[num_trials-1, task]
                    b_e = losses[model_name]

            print("BE: {}".format(b_e))
            losses['Best Expert'].loc[:,task] = b_e.loc[:,task]

        
        #Print Losses Graphs  
        fig = plt.figure(figsize=(8.0, 5.0))
        ax = plt.axes()
        
        #Plot learner and model losses
        for key, learner in enumerate(learners):
            ax.plot(losses[learner_names[key]], linewidth=1.0)
            ax.legend(losses[learner_names[key]].columns, bbox_to_anchor=(1.05, 1), loc='upper left')

            ax.set_title("(k:"+str(switchval)+", experts:"+str(num_experts)+", eta:"+str(eta)+" ,c:" +str(c)+")",fontsize=16)
            fig.suptitle(learner_names[key]+" Loss" , fontsize=24, y=1)

            ax.set_xlabel('Year')
            ax.set_xticks([i*12 for i in range(0,11)])
            ax.set_xticklabels([1990+i for i in range(0,11)])
            ax.set_ylabel('Squared loss')
            plt.savefig(learner_names[key]+" Loss, task:"+str(task)+"(k="+str(switchval)+")", dpi=1000, bbox_inches='tight')
            plt.show()
 

        #Plot Predictions
        for task in range(10):
            fig = plt.figure(figsize=(18.0, 15.0))
            ax = plt.axes()
            ave = np.zeros(120)
            for name in models.keys():
                model = models[name]
                #ax.plot(np.arange(120),np.asarray(prediction[name].loc[:, task].values), label=name)
                ave = ave + np.asarray(prediction[name].loc[:, task].values)
            ave = ave / (len(models.keys()))
            ax.plot(np.arange(120),np.asarray(ave), label="Average GCM Value", linewidth=2)
            ax.plot(np.arange(120),np.asarray(swarm_predictions.loc[:, task].values), label="Swarm", linewidth=2, zorder=5)
            ax.plot(np.arange(120),np.asarray(ground_val.loc[:, task].values), label="Actual Value (NASA GISTEMP)", linewidth=2)
            ax.set_title("(k:"+str(global_switches)+", experts:"+str(num_experts)+", eta:"+str(eta)+" ,c:" +str(c)+")",fontsize=16)
            fig.suptitle("Zone " +str(task)+" Anomaly Predictions", fontsize=24, y=1)
            ax.set_xlabel('Year')
            ax.set_xticks([i*12 for i in range(0,11)])
            ax.set_xticklabels([1990+i for i in range(0,11)])
            ax.set_ylabel('Anomaly Prediction (C)')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.savefig("Zonal Anomaly Predictions (k="+str(global_switches)+")"+str(task)+str(eta), bbox_inches='tight')
 
        #Plot Weights
        for key, learner in enumerate(learners):
            for task in range(learner.s):
                fig = plt.figure(),
                ax = plt.axes()
                thisplot = ax
                thisplot.plot(weights[learner_names[key]][task], linewidth=1.0)
                thisplot.set(xlim=(0, num_trials))
                thisplot.set_xticks([i*12 for i in range(0,11)])
                thisplot.set_xticklabels([1990+i for i in range(0,11)])
                thisplot.legend(["Expert " + str(i+1) for i in range(modes)], loc=6, bbox_to_anchor=(1.05, 1), loc='upper left')
                thisplot.set_title("(k:"+str(switchval)+", experts:"+str(num_experts)+", eta:"+str(eta)+" ,c:" +str(c)+")",fontsize=16)
                fig.suptitle(learner_names[key]+", location "+str(task)+" weights", fontsize=24, y=1)
                thisplot.set_xlabel('Year')
                thisplot.set_ylabel('Weights')
                plt.savefig(learner_names[key]+", location "+str(task)+" weights, k="+str(switchval))
                #plt.show()

