import numpy as np
import copy
import random as rd
import pandas as pd
from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from func import Tree,mcts_mcts
import matplotlib.pyplot as plt
import time
import tqdm

np.random.seed(11)

ngames=500
r=[Rating(),Rating(),Rating(),Rating()]
r_result=np.zeros((ngames,4))

start1=time.time()
boardSize=4

# #experiments on cp
# cp=np.array([0.1,0.5,1,1.5])
# #N=10,100,500: choose a value each time
# N=500
# for i in tqdm.tqdm(list(range(ngames))):
#     num=rd.sample(range(4),2)
#     n1=num[0]
#     n2=num[1]
#     result=mcts_mcts(cp1=cp[n1], N1=N,cp2=cp[n2], N2=N, size=boardSize,print_all=False,first=True)
#     if result:
#        r[n1],r[n2]=rate_1vs1(r[n1],r[n2])
#     else:
#         r[n2],r[n1]=rate_1vs1(r[n2],r[n1])
#     #update
#     for j in range(4):
#         r_result[i,j]=r[j].mu
#
# np.savetxt('1N500cp.txt',r_result)

#experiments on N：
##cp=0.1,0.5,1,1.5, choose a value each time
# cp=0.1
# N=np.array([10,100,500,1000])
# for i in tqdm.tqdm(list(range(ngames))):
#     num=rd.sample(range(4),2)
#     n1=num[0]
#     n2=num[1]
#     result=mcts_mcts(cp1=cp, N1=N[n1],cp2=cp, N2=N[n2], size=boardSize,print_all=False,first=True)
#     if result:
#        r[n1],r[n2]=rate_1vs1(r[n1],r[n2])
#     else:
#         r[n2],r[n1]=rate_1vs1(r[n2],r[n1])
#     #update
#     for j in range(4):
#         r_result[i,j]=r[j].mu
# #saveout result
# np.savetxt('1Ncp0.1.txt',r_result)

#plot experiment1
# cp=np.array([0.1,0.5,1,1.5])
# plt.figure(figsize=(8,4))
# plt.subplot(1,3,1)
# y=np.loadtxt('1N10cp.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='cp={:}'.format(cp[i]))
# plt.title('N=10')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc='upper right')
# plt.subplot(1,3,2)
# y=np.loadtxt('1N100cp.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='cp={:}'.format(cp[i]))
# plt.title('N=100')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc='upper right')
# plt.subplot(1,3,3)
# y=np.loadtxt('1N500cp.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='cp={:}'.format(cp[i]))
# plt.title('N=500')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc='upper right')
# plt.tight_layout()
# plt.savefig('mcts_para_cp.png',dpi=150)

#plot experiment2
# N=np.array([10,100,500,1000])
# plt.figure(figsize=(7,5))
# plt.subplot(2,2,1)
# y=np.loadtxt('1Ncp0.1.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='N={:}'.format(N[i]))
# plt.title('cp=0.1')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc=8,prop={'size': 6})
# plt.subplot(2,2,2)
# y=np.loadtxt('1Ncp0.5.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='N={:}'.format(N[i]))
# plt.title('cp=0.5')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc=8,prop={'size': 6})
# plt.subplot(2,2,3)
# y=np.loadtxt('1Ncp1.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='N={:}'.format(N[i]))
# plt.title('cp=1')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc=8,prop={'size': 6})
# plt.subplot(2,2,4)
# y=np.loadtxt('1Ncp1.5.txt')
# for i in range(4):
#     plt.plot(range(ngames),y[:,i],label='N={:}'.format(N[i]))
# plt.title('cp=1.5')
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend(loc=8,prop={'size': 6})
# plt.tight_layout()
# plt.savefig('mcts_para_N.png',dpi=150)
