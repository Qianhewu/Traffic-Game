#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import queue
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns
class traffic_game():
    def __init__(self, N, f, M, AVG, VAR):
        self.N = N
        self.f = f
        self.M = M
        self.AVG = AVG
        self.VAR = VAR
        self.theta =  np.minimum(np.maximum(np.random.normal(self.AVG,self.VAR,self.N),0.2),0.8)
#         self.t = np.random.uniform(0,self.theta, self.N)
        self.t = self.theta - 0.2
    
    def updateNf(self, N, f):
        self.N = N
        self.f = f
        
    def init_t(self):
        self.t = self.theta - 0.2
        
    def runs(self, t):
        finished_t = np.zeros(self.N)
        queuing_n = 1
        finished = 0
        sorted_t = np.sort(t)
        index = np.argsort(t)
        q = queue.Queue()
        current = sorted_t[0]
        q.put(index[0])
        x = [0,current, current]
        y = [0,0,1]
        while finished < self.N:
            if q.qsize() > 0:
                current += 1/self.f
                finished += 1
                finished_t[q.get()] = current
                for i in range(queuing_n,self.N):
                    if sorted_t[i] <= current:
                        q.put(index[i])
                        queuing_n += 1
                x.append(current)
                y.append(q.qsize())
            else:
                x.append(current)
                y.append(q.qsize())
                if queuing_n >= self.N:
                    break
                current = sorted_t[queuing_n]
                q.put(index[queuing_n])
                queuing_n += 1
                x.append(current)
                y.append(q.qsize())
        return x,y,finished_t
    
    def compute_optimal(self):
        epsilon = 0.001
        opt = np.zeros(len(self.t))
        for i in range(len(opt)):
            c = self.t.copy()
            a = 0
            b = 1
            while b-a > epsilon:
                c[i] = (a+b)/2
                x,y,finish = self.runs(c)
                if finish[i]>self.theta[i]:
                    b = (a+b)/2
                else:
                    a = (a+b)/2
            opt[i] = (a+b)/2
        return self.theta - opt
    
    def cost(self, step):
        epsilon = 0.01
        x,y,ft = self.runs(self.t)
        aot = self.theta - self.t
        faot = self.theta - ft
        cost = aot * (faot>=0) + (faot < 0)*self.M
        update = faot/(step+1)*(faot>epsilon) + (faot<0)*faot + (faot < epsilon)*(faot - epsilon)/(step+1)
        return cost, update
    
    def update(self, gradient):
        self.t = np.minimum(np.maximum(self.t + gradient,0),1)
        
    def info(self):
        return self.theta, self.t
    
class detour_game():
    def __init__(self, N, f, arr):
        self.N = N # number of stations
        self.f = f
        self.arr = arr
        self.interval = 0.001
        self.queue = []
        self.strategy = np.ones([4,N])
        self.strategy[3][0] = 0
        self.current = 0
        self.train = []
        self.total = 0
        self.finish = 0
        self.time = 0
        self.update_cache = np.zeros([4,N])
        self.update_cache_num = np.zeros([4,N])
        self.update_num = 0
        for i in range(self.N):
            self.queue.append([])
            self.train.append([0,0,0,0])
            
    def step(self):
        arr_prob = self.arr * self.interval
        t = self.current
        for i in range(int(1/(self.f*self.interval))):
            self.current += self.interval
            r = random.uniform(0,1)
            if r < arr_prob:
                self.total += 1
                length = len(self.queue[0])
                r = random.uniform(0,1)*self.strategy[length].sum()
                for i in range(self.N):
                    if r >= self.strategy[length][i]:
                        r -= self.strategy[length][i]
                    else:
                        self.queue[i].append([self.current,length,i])
                        break
        self.current = t + 1/self.f
        for i in range(self.N):
            if self.train[i][0] != 0:
                continue
            if self.train[i][0] == 0 and len(self.queue[i]) > 0:
                if self.current >= self.queue[i][0][0] + i: # in-queue time
                    self.train[i][0] = self.queue[i][0][0] # join time
                    self.train[i][1] = self.current + i # finish time
                    self.train[i][2] = self.queue[i][0][1] # strategy:length
                    self.train[i][3] = self.queue[i][0][2] # strategy:station
                    self.queue[i] = self.queue[i][1:]
        start = self.train[0][0]
        finish = self.train[0][1]
        strategy_length = self.train[0][2]
        strategy_station = self.train[0][3]
        if start != 0:
#             print("start, finish", self.train[0])
            self.train = self.train[1:]
            self.train.append([0,0,0,0])
            self.finish += 1
            self.time += finish - start
            self.update_cache[strategy_length][strategy_station] += finish - start
            self.update_cache_num[strategy_length][strategy_station] += 1
            self.update_num += 1
        return start, finish
    
    def update(self):
        avg = self.time/self.update_num
#         print(avg)
        epsilon = 0.1
        for i in range(4):
            for j in range(self.N):
                if self.update_cache[i][j] != 0:
                    c = self.update_cache[i][j] / self.update_cache_num[i][j] - avg
                    self.update_cache[i][j] = 0
                    self.update_cache_num[i][j] = 0
                    self.strategy[i][j] *= pow((1-epsilon),c)
        return avg        
                    
    def print_prob(self):
        print("prob:", self.strategy[0]/self.strategy[0].sum())
        print("prob:", self.strategy[1]/self.strategy[1].sum())
        print("prob:", self.strategy[2]/self.strategy[2].sum())
        print("prob:", self.strategy[3]/self.strategy[3].sum())
    
    def print_queue(self):
        print("queue length:")
        for i in range(self.N):
            print(len(self.queue[i]))
        print("queue",self.queue)
                    

def visualize(x,y,path = None, marksize = None, xl = None, yl = None,mode = None, e = None):
#     plt.plot(x, y, 'ro-', color='#4169E1', alpha=0.8, linewidth=1)
    if mode == "variance":
        plt.figure(figsize=(6.4,4.8), dpi=200)
        sns.tsplot(time=x, data=y, color = 'b', linestyle='-')
        return
    if mode == "errorbar":
        plt.figure(figsize=(6.4,4.8), dpi=200)
        plt.errorbar(x, y, yerr = e, linestyle='-', marker = 'o',markersize= marksize)
    if mode == "exhibit":
        plt.figure(figsize=(6.4,4.8), dpi=200)
    if marksize != None:
        plt.plot(x, y, 'ro-',markersize = marksize, color='#4169E1', alpha=0.8, linewidth=1)
    else:
        plt.plot(x, y, color='#4169E1', alpha=0.8, linewidth=1)
    if xl != None:
        plt.xlabel(xl)
    if yl != None:
        plt.ylabel(yl)
    if path != None:
        plt.savefig(path)
    plt.show()
    

def no_detour_simulate():
    game = traffic_game(10,20,1,0.6,0.2)
    game.init_t()
    game.updateNf(10,20)
    NUM_DAYS = 50000
    c = []
    ns = []
    cmin = 100
    nsmin = 100
    for i in range(NUM_DAYS):
    	cost, gradient = game.cost(i)
    	c.append(cost.sum())
#     	opt = game.compute_optimal()
#     	ns.append(np.array(cost - opt).mean())
#     	if nsmin > np.array(cost - opt).mean():
#     	    nsmin = np.array(cost - opt).mean()
#     	if np.array(cost - opt).mean() < 0.01 and i > 500:
#     	    cmin = cost.sum()
#     	    break
    	game.update(gradient)

	z = []
    for i in range(99,len(c)):
        d = np.array(c[max(0,i-99):i+1]).mean()
        c[i] = d
    	z.append(i)
    c = c[99:]
    visualize(z,c, path = "cost-10-20",xl = "time", yl = "total cost")
    print(nsmin, cmin)




def detour_simulate():
    detour = detour_game(N = 5, f = 20, arr = 19.5)
    MAX_DAYS = 150000
    last_avg = 0
    avg = []
    x = []
    for i in range(MAX_DAYS):
    	detour.step()
    	if i % 100 == 99:
    	    avg.append(detour.update())
    	    x.append(i)
    visualize(x,avg,"./Figure/detour_converge", xl = "iters",yl = "cost")
    print("final:",avg[-1])
#         if abs(avg - last_avg)<0.0001 and i >= MAX_DAYS/2:
#             break


# In[226]:


# x = [0.5, 0.75, 0.9, 0.95, 0.96, 0.97, 0.98, 0.985]
# t = []
# color = ["#000000","#dc143c","#8a2be2","#0000ff","#7fff00","#6495ed", "#c17585","#ffd700"]
# for i in range(1500):
#     t.append(i*100)
# plt.figure(dpi = 200)
# for i in range(8):
#     plt.plot(t,data[i],color = color[i],markersize = 2,linestyle = '-',linewidth = 1,label = str(x[i]))
# plt.legend(loc = [1.01,0])
# plt.savefig("convergence-detour",bbox_inches = 'tight')
# plt.show()



