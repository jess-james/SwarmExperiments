import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from collections import defaultdict, Counter
from heapq import heappush,heappop

#Loss Functions
def L(y, y_hat):
    return (y-y_hat)**2

def Llog(y, y_hat):
    if len(yhat>1):
        c = np.zeros(len(yhat))
        for i in range(0,len(yhat)):
            c[i] = -y*np.log(yhat[i]) - (1-y)*np.log(1-yhat[i])
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
        self.W = [np.ones(n_experts)/n_experts for _ in range(tasks)]

    def update(self, x, y, task):
        self.W[task] *= np.exp(-self.eta*L(y,x))

    def predict(self, x, task):

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

    def update(self, x, y,task):
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
        #self.eta = np.sqrt((m*np.log(n/m)+s*m*n*(1/m)+(T-s)*n*(k/(T-s))+(m-1)*(T-s)*n*(k/((m-1)*(T-s))))*(2/T))
        self.eta = eta
        self.c = 1/eta

    def predict(self, x, task):
        '''v = (self.pi*self.W[task])/(self.pi @ self.W[task])
        d0 = -self.c*np.log(np.exp(-self.eta*L(0,x)) @ v)
        d1 = -self.c*np.log(np.exp(-self.eta*L(1,x)) @ v)
        return 0.5*(np.sqrt(d0) + 1 - np.sqrt(d1))'''
        
        v = (self.pi*self.W[task])/(self.pi @ self.W[task])
        return v @ x

    def update(self, x, y, task):
        c = L(y, x)
        delta = self.W[task]*np.exp(-self.eta*c)
        beta = (self.pi@self.W[task])/(self.pi@delta)
        epsilon = 1 - self.W[task] + beta*delta
        self.pi = self.pi*epsilon
        self.W[task] = (self.phi*(1-self.W[task]) + self.theta*beta*delta)*(1/epsilon)

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



class experiment:
    def __init__(self, learners, learner_names, experiment_params, experiment_type=None):
        self.learners = learners
        self.learner_names = learner_names
        self.s, self.n, self.m, self.k, self.T = experiment_params
        self.local_t = np.zeros(self.s, np.int64)
        self.local_k = np.zeros(self.s, np.int64)

    def expert_matrix(self):
        experts = [[] for i in range(self.s)]
        if self.m == 1:
            for task in range(self.s):
                experts[task] = [0 for i in range(int(self.k/self.s)+1)]
        elif self.m ==2:
            for task in range(self.s):
                start = np.random.randint(0,2)
                experts[task] = [(start+i)%2 for i in range(int(self.k/self.s)+1)]
        else:
            experts = rearrange(self.m, self.k, self.s)
        trial_length = self.T/self.s
        print(f"tl:{trial_length}")
        seg_length = int(trial_length/(1+(self.k/self.s)))
        for ind, exp in enumerate(experts):
            new_exp = []
            for i in exp:
                for t in range(int(seg_length)):
                    new_exp.append(i)
            while len(new_exp)<trial_length:
                new_exp.append(experts[ind][-1])
            experts[ind] = new_exp
            print(f"exp: {len(new_exp)}")
        experts = np.asarray(experts)
        return experts

#for index, task_mode_list in enumerate(task_modes):
#    print(task_mode_list)
#    unchecked = True
#    while unchecked:
#        random.shuffle(task_mode_list)
#        for x in range(len(task_mode_list)-1):
#            if task_mode_list[x] == task_mode_list:
#                if index==(len(task_modes)-1):
#                    return expert_matrix()
#                continue
#        unchecked = False
#    experts[index] = task_mode_list
    #print(task_mode_list)

    def experiment_data(self, uniform=True, p=0):
        per_task_switches = int(self.k/self.s)
        switches = dict()
        experts = dict()
        if uniform:
            #Assuming a uniform distribution across modes
            experts = self.expert_matrix()
            for task in range(self.s):
                task_length = int(self.T/self.s)
                trial_length = int(task_length / (per_task_switches+1))
                switches[task] = [i* trial_length for i in range(per_task_switches+1)]
        else:
            modes = [i for i in range(self.m)]
            experts = [[] for i in range(self.s)]
            task_length = int(self.T/self.s)
            for task in range(self.s):
                task_experts = [np.random.choice(modes)]
                task_switches = [0]
                for local_time in range(1, task_length):
                    switch_bool = np.random.choice([0,1], p=[1-p, p])
                    if switch_bool:
                        task_switches.append(local_time)
                        task_experts.append(np.random.choice([x for x in modes if x!=task_experts[-1]]))
                    else:
                        task_experts.append(task_experts[-1])
                switches[task] = task_switches
                experts[task] = np.asarray(task_experts)
        experts = np.asarray(experts)
        task_vector = []
        for t in range(int(self.T/self.s)):
            for u in range(self.s):
                task_vector.append(u)
        random.shuffle(task_vector)
        self.switches = switches
        self.experts = experts
        self.task_vector = task_vector
        return switches, experts, task_vector

    def environment_data(self, switches, experts, task_vector):
        X = np.random.uniform(0, 0.5, (self.T, num_experts))
        Y = np.random.randint(0,2, self.T)
        local_t = np.zeros(self.s, np.int64)
        local_mode = np.zeros(self.s, np.int64)
        for t in range(self.T):
            X[t] = Y[t]*(1-X[t]) + (1-Y[t])*(X[t])
            current_task = task_vector[t]
            current_t = local_t[current_task]
            local_t[current_task] +=1
            #current_mode = local_mode[current_task]
            #if(current_t in switches[current_task]) and not current_t==0:
            #        local_mode[current_task] +=1
            current_expert = experts[int(current_task)][current_t]
            #print(f"Current task: {current_task}, local_time: {current_t}, mode: {current_mode}")
            X[t, current_expert] = Y[t] * np.random.uniform(0.85, 1) + (1-Y[t])*np.random.uniform(0, 0.15)
        return X,Y

    def step(self,tau):
            task = self.task_vector[tau]
            t = self.local_t[task]
            #print(str(task)+":"+str(t))
            if t in switches[task] and t!=0:
                self.local_k[task]+=1
            k = self.local_k[task]
            this_expert = self.experts[task][k]
            self.local_t[task] += 1
            return task, t, k, this_expert #and weights



def run_experiments(s, m, k, eta):
        #Experiment set up
        tasks = s
        modes = m
        p = k if 0<k<1 else 0
        global_switches = global_trials*p if p>0 else global_switches
        global_trials = 1000 * tasks
        num_experts = 64
        eta = eta
        c = 1/eta
        
        num_trials = global_trials/tasks
        uniform=True
        v_alpha = (global_switches/tasks)/(2*(global_switches/tasks) + 50)
        f_alpha = (global_switches/tasks)/(num_trials-1)
        expert_list = np.arange(modes)
        swarm_learner = swarm(tasks, num_experts, modes, global_switches, global_trials, eta)
        static_learner = static_expert_learner(num_experts, eta, c, tasks)
        fixed_share = fixed_share_learner(num_experts, eta, c, f_alpha, tasks)
        var_share = variable_share_learner(num_experts, eta, c, v_alpha, tasks)
        learners = [swarm_learner, fixed_share, var_share]
        learner_names = ['Swarm', 'Fixed Share', 'Variable Share', 'Best Partition', 'Average Expert', 'Best Expert']
        basic_experiment = experiment(learners, learner_names, (tasks,num_experts,modes,global_switches,global_trials))
        switches, experts, task_vector = basic_experiment.experiment_data(uniform=uniform)#False, p=p)
        X, Y = basic_experiment.environment_data(switches, experts, task_vector)
        weights = {}
        losses = {}

        exploss = [pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials)) for _ in range(num_experts)]
        for learner_name in learner_names:
            losses[learner_name] = pd.DataFrame(0.0, columns = np.arange(tasks), index = np.arange(num_trials))
            if not learner_name == "Best Partition" and not learner_name == "Average Expert":
                weights[learner_name] = {}
                for i in range(tasks):
                    weights[learner_name][i] = pd.DataFrame(0.0, columns = np.arange(num_experts), index = np.arange(num_trials))
        learner_loss = np.zeros(len(learners), np.float64)
        task_loss = [[] for i in range(tasks)]

        #Run experiments
        for tau in range(global_trials):
            task, t, k, expert = basic_experiment.step(tau)#, X[tau], Y[tau])
            for key, learner in enumerate(learners):
                y_hat = learner.predict(X[tau], task)
                learner_loss[key] += np.sum(L(Y[tau], y_hat)) #fix
                learner.update(X[tau],Y[tau],task)

                losses[learner_names[key]].loc[t, task] = losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                weights[learner_names[key]][task].iloc[t,:] = learner.W[task]/np.sum(learner.W[task])

            if t==0:
                losses['Average Expert'].loc[t, task] = np.sum(L(X[tau, task], Y[tau]))/num_experts #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                #losses['Average Expert'].loc[t, task] = losses['Average Expert'].loc[t-1, task] + (np.mean([L(X[tau][i],Y[tau]) for i in range(num_experts)])) #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                losses['Best Partition'].loc[t, task] = np.min([L(X[tau][i],Y[tau]) for i in range(num_experts)])
                for exp in range(num_experts):
                    exploss[exp].loc[t,task] = L(X[tau][exp], Y[tau])
            else:
                losses['Average Expert'].loc[t, task] = losses['Average Expert'].loc[t-1, task] + np.sum(L(X[tau], Y[tau]))/num_experts #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                #losses['Average Expert'].loc[t, task] = losses['Average Expert'].loc[t-1, task] + (np.mean([L(X[tau][i],Y[tau]) for i in range(num_experts)])) #losses[learner_names[key]].loc[t-1, task] + L(Y[tau], y_hat) if t>0  else L(Y[tau], y_hat)
                losses['Best Partition'].loc[t, task] = losses['Best Partition'].loc[t-1, task] + np.min([L(X[tau],Y[tau]) for i in range(num_experts)])
                for exp in range(num_experts):
                    exploss[exp].loc[t,task] = L(X[tau][exp], Y[tau]) + exploss[exp].loc[t-1,task]

        #Plot Losses
        fig = plt.figure()
        ax = plt.axes()
        colourmap = plt.cm.get_cmap('Set3', modes)
        if p==0:
            if tasks > 1:
                ax.set_title(f"{tasks} tasks, {global_switches} global switches, {modes} modes")
            else:
                ax.set_title(f"{tasks} task, {global_switches} global switches, {modes} modes")

        else:
            ax.set_title(f"{tasks} tasks, uniform switching, {modes} modes")
            #ax.set_title(f"{tasks} tasks, switch with probability {p}, pool of {modes}")
        ax.set_xlabel('Local Time')
        ax.set_ylabel('Task Number')
        psm = ax.pcolormesh(experts+1, cmap=colourmap, rasterized=True, vmin=0.5, vmax=modes+0.5)
        fig.colorbar(psm, ax=ax, ticks=[i+1 for i in range(modes)], label="Best Expert")
        ax.set_yticks([i+0.5 for i in range(tasks)])
        ax.set_yticklabels([i+1 for i in range(tasks)])
        plt.savefig('figs/colourmesh'+str(expnum)+'.png')
        #plt.show()
        
        #Plot Loss per Task
        for task in range(tasks):
            fig = plt.figure()
            ax = plt.axes()
            for key, learner in enumerate(learners):
                plt.plot(losses[learner_names[key]].loc[:,task].values, linewidth=1.0, label=learner_names[key])
                #plt.set(xlim=(0, num_trials+10))
            expres= [tab.loc[num_trials-1,task] for tab in exploss]
            bestexind = expres.index(min(expres))   
            plt.plot(exploss[bestexind].loc[:,task], linewidth=1.25, label='Best Expert')
            plt.xticks((switches[task]), list(range(1, switches+1)))#[experts[task][i]+1 for i in switches[task]] )#([i*int(global_trials/(tasks+global_switches)) for i in range(int(num_trials/(global_trials/(tasks+global_switches))))])
            plt.title(f"Task {task} Loss")
            plt.xlabel('Segment')
            plt.ylabel('Squared loss')
            plt.legend()
            plt.savefig('figs/exp'+str(expnum)+'task'+str(task)+'.png')

        #Plot overall loss
        expres= [np.sum(tab.loc[num_trials-1,:]) for tab in exploss]
        bestexind = expres.index(min(expres))
        losses['Best Expert'] = exploss[bestexind]

        fig = plt.figure()
        ax = plt.axes()
        for key, learner in enumerate(learners):
            plt.plot(losses[learner_names[key]].sum(axis=1), linewidth=1.25, label=learner_names[key])
        plt.plot(losses['Best Partition'].sum(axis=1), linewidth=1.25, label='Best Partition')
        plt.plot(losses['Average Expert'].sum(axis=1), linewidth=1.25, label='Average Expert')
        plt.plot(losses['Best Expert'].sum(axis=1), linewidth=1.25, label='Best Expert')

        plt.xticks(switches[0][1:], list(range(1,(global_switches//tasks)+1)))#([i*int(global_trials/(tasks+global_switches)) for i in range(int(num_trials/(global_trials/(tasks+global_switches))))])
        plt.xlabel('Segment')
        plt.ylabel('Squared loss')
        plt.legend()
        plt.savefig('figs/exp'+str(expnum)+'.png')
       #plt.show()


    #Plot Weights
    fig, ax = plt.subplots(2, 5)
    fignum = 0
    fig.suptitle("Swarm Weights")
    for key, learner in enumerate(learners):
        if learner_names[key] == "Swarm":
            for task in range(tasks):
                x = (fignum//5)
                y = (fignum%5)
                thisplot = ax[x,y]
                thisplot.plot(weights[learner_names[key]][task], linewidth=1.0)
                thisplot.set(xlim=(0, num_trials))
                thisplot.set_xticks(switches[task])
                thisplot.set_xticklabels([experts[task][i]+1 for i in switches[task]])
                thisplot.set_title(learner_names[key]+": task "+str(task))
                thisplot.set_xlabel('Mode')
                thisplot.set_ylabel('Weights')
                fignum+=1
        fig.legend(["Expert " + str(i+1) for i in range(modes)])
        plt.savefig(f'figs/weights{learner_names[key]}')
        plt.show()
    #df.plot(subplots=True, layout=(4,5))

tasks = [10]
switches = [0, 10, 100]
modes = [10, 5 ,2]
etas = [2] #As in Learning the Best Expert

for s in tasks:
    for m in modes:
        for k in switches:
            for eta in etas:
                run_experiments(s=s, m=m,k=k,  eta=eta)

