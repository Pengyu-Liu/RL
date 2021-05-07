import numpy as np
import copy
import random as rd
import pandas as pd
from trueskill import Rating, quality_1vs1, rate_1vs1
from hex_skeleton import HexBoard
from func import Tree,xy_read,human_compu,alphabeta,alphabeta_randomVSdijkstra, mcts_alphabeta, idtt_alphabeta, mcts_mcts
import matplotlib.pyplot as plt
import time
import tqdm

np.random.seed(11)
#human vs computer: if you want to test one of the cases: uncomment the corresponding two lines

# print('alphabeta algorithm, random, depth=',3)
# human_compu('alphabeta','random',3)

# print('alphabeta algorithm, dijkstra, depth=',3)
# human_compu('alphabeta','dijkstra',3)
#
# print('alphabeta algorithm, dijkstra, depth=',4)
# human_compu('alphabeta','dijkstra',4)
#
# print('\n idtt algorithm, random, depth=',5)
# human_compu('idtt')
#
# print('\n mcts algorithm')
# human_compu('mcts')

#---------------------------------------------------------------------------------------------------
#computer VS computer
ngames=50
#rating for method1
r1=Rating()
#rating for method2
r2=Rating()
r1_result=np.zeros((ngames,3))
r2_result=np.zeros((ngames,3))
start1=time.time()
boardSize=3
#two paramters for mcts
cp=1
N=100
#time control for idtt
time_limit=5
idtt_depth=5
#start play games
for i in tqdm.tqdm(list(range(ngames))):
    #3 experiments using alphabeta with random or dijkstra evulation: method can choose 'random' and 'dijkstra'
    #uncomment following code if choose alphabeta VS alphabeta: random and dijkstra; dijkstra and dijkstra
    # if i%2==0:
    #     #method1 moves first
    #    result=alphabeta_randomVSdijkstra(method1='dijkstra',method2='dijkstra',depth1=3,depth2=4,size=boardSize,print_all=False,first=True)
    # else:
    # #method2 moves first
    #    result=alphabeta_randomVSdijkstra(method1='dijkstra',method2='dijkstra',depth1=3,depth2=4,size=boardSize,print_all=False,first=False)

#----------------------------------------------------------------------------------------------------------------------------------------------
   # experimnets using idtt and alphabeta
    #uncomment if choose idtt VS alphabeta (meethod can choose 'random' or 'dijkstra')
    # if i%2==0:
    #     #method1 moves first
    #    result=idtt_alphabeta(method='dijkstra',idtt_depth=idtt_depth,depth2=3,size=boardSize,print_all=False,first=True,time_limit=time_limit)
    # else:
    # #method2 moves first
    #    result=idtt_alphabeta(method='dijkstra',idtt_depth=idtt_depth,depth2=3,size=boardSize,print_all=False,first=False,time_limit=time_limit)
#

#---------------------------------------------------------------------------------------------------------------------------------------------
#     # experiments using mcts and alphabeta
#    #uncomment if choose mcts VS alphabeta
    # if i%2==0:
    #     #method1(mcts) moves first
    #     result=mcts_alphabeta(cp=cp, N=N,method='dijkstra',depth=3,size=boardSize,print_all=False,first=True)
    # else:
    # #method(alphabeta) moves first
    #     result=mcts_alphabeta(cp=cp, N=N,method='dijkstra',depth=3,size=boardSize,print_all=False,first=False)


#----------------------------------------------------------------------------------------------------------------------------------------------
#    #experiments using mcts aand idtt
#   #uncomment if choose mcts VS idtt
    # if i%2==0:
    #     #method1(mcts) moves first
    #     result=mcts_alphabeta(cp=cp, N=N,method='idtt',depth=idtt_depth,size=boardSize,print_all=False,first=True,time_limit=time_limit)
    # else:
    # #method(alphabeta) moves first
    #     result=mcts_alphabeta(cp=cp, N=N,method='idtt',depth=idtt_depth,size=boardSize,print_all=False,first=False,time_limit=time_limit)


#update result after each game
    if result:
       #method1 win
       r1,r2=rate_1vs1(r1,r2)
       #update
       r1_result[i,0]=r1.mu
       r1_result[i,1]=r1.sigma
       r1_result[i,2]=1
       r2_result[i,0]=r2.mu
       r2_result[i,1]=r2.sigma
    else:
        #method2 win
        r2,r1=rate_1vs1(r2,r1)
        #update
        r1_result[i,0]=r1.mu
        r1_result[i,1]=r1.sigma
        r2_result[i,0]=r2.mu
        r2_result[i,1]=r2.sigma
        r2_result[i,2]=1

end1=time.time()
print('running time is: {:.2f} s'.format(end1-start1))
# plt.figure(figsize=(7,5))
# plt.subplot(1,2,1)
# plt.plot(range(ngames),r1_result[:,0],label='r1')
# plt.plot(range(ngames),r2_result[:,0],label='r2')
# plt.xlim(0,50)
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend()
# plt.subplot(1,2,2)
# plt.errorbar(range(ngames),r1_result[:,0],yerr=r1_result[:,1],label='r1')
# plt.errorbar(range(ngames),r2_result[:,0],yerr=r2_result[:,1],label='r2')
# plt.xlim(0,50)
# plt.xlabel('number of games')
# plt.ylabel('Elo rating')
# plt.legend()
# plt.tight_layout()
# plt.savefig('mcts100_dij3.png',dpi=150)
print('Final elo_rating for method1:{:.3f}'.format(r1.mu))
print('Final elo_rating for method2:{:.3f}'.format(r2.mu))
print('win times for method1:{:d}'.format(int(np.sum(r1_result[:,2],axis=0))))
print('win times for method2:{:d}'.format(int(np.sum(r2_result[:,2],axis=0))))
