import numpy as np
import pandas as pd
import copy
import time

from hex_skeleton import HexBoard

#--------------------------------------------------------------------------------
class Tree:
    #node tree
    def __init__(self,Hexboard,color,parent=None,coordinate=None,my_color=None):
        #currrent state
        self.state=Hexboard
        self.parent=parent
        self.children=[]
        self.untriedmove=self.state.getMoveList()
        self.visit=0
        self.win=0
        self.location=coordinate
        #opponent's color
        self.color=color
        #used for initization of alphabeta
        if color==my_color:
            self.gvalue=-9999
        else:
            self.gvalue=9999

    # add children under tree structure in alphabeta algorithm
    def add_child_ab(self):
        for move in self.untriedmove:
             # add all empty places into children
            child_state=copy.deepcopy(self.state)
            child_color=self.state.get_opposite_color(self.color)
            child_state.place(move,child_color)
            child_node=Tree(child_state,parent=self,coordinate=move,color=child_color)
            self.children.append(child_node)

    # add children under tree structure in monte-carlo tree search algorithm
    def add_child(self):
        #randomly choose an unexpanded child
        pick=np.random.randint(len(self.untriedmove))
        #pick this child and remove it from untriedmove
        child= self.untriedmove.pop(pick)
        child_color=self.state.get_opposite_color(self.color)
        child_state=copy.deepcopy(self.state)
        child_state.place(child,child_color)
        child_node=Tree(child_state,parent=self,coordinate=child,color=child_color)
        self.children.append(child_node)
        return child_node

    def is_fully_expanded(self):
        return len(self.untriedmove)==0

    def is_terminal_node(self):
        return self.state.is_game_over()
#--------------------------------------------------------------------------------
#--------------------------text interface----------------------------------------
def xy_read(Game):
    """text interface: read (x,y), accept it only if it is the empty place and
    within the board size range"""
    x = int(ord(input(' x(a,b,c...): '))-97)
    y = int(input(' y(0,1,2...): '))

    while x>Game.size or y>Game.size:
        print('Invaild place, input again!!!')
        x = int(ord(input(' x(a,b,c...): '))-97)
        y = int(input(' y: '))
    while not Game.is_empty((x,y)):
        print('Invaild place, input again!!!')
        x = int(ord(input(' x(a,b,c...): '))-97)
        y = int(input(' y: '))
    return x,y
#--------------------------------------------------------------------------------
#-------------------------human vs computer experiment---------------------------
def human_compu(algor,method='random',depth=3):
    """A function that human play with computer, user can choose their prefered
    board size(3,4 recommanded), color, want to take first move"""
    global flag,INF, n_nodes, cutoff, Game, size

    print('Choose a board size:')
    size=int(input())
    Game=HexBoard(size)
    print('Choose a color: 1(BLUE) or 2(RED)')
    print('Blue: from left to right; Red: from top to bottom')
    opp_color=int(input())
    my_color=Game.get_opposite_color(opp_color)
    print('Do you want to start first? Yes(y) or No(n)')
    first=input()
    print('Game start!')
    # human first move do or not
    if first=='y' or first=='Yes':
        Game.print()
        x,y = xy_read(Game)
        Game.place(coordinates=(x,y),color=opp_color)
        Game.print()
        root=Tree(Game,color=opp_color,parent=None,coordinate=(x,y),my_color=my_color)

    else:
        first_color = my_color
        last_color = opp_color
        root=Tree(Game,color=opp_color,parent=None,coordinate=None,my_color=my_color)

    INF = 99999  # sufficient large number

    # human and computer play the game until one of them win
    while not Game.is_game_over():
        if algor=='alphabeta':
            # varibales intialization
            n_nodes = 0
            cutoff = 0
            flag = 0
            my_move=alphabeta(n=root, a=-INF, b=INF, d=depth, method=method,depth=depth,my_color=my_color,opp_color=opp_color)
            print('n_nodes=',n_nodes,'\n cutoff=',cutoff)

        elif algor=='idtt':
            flag = 0
            transpositiontable = []
            n_nodes = 0
            cutoff = 0
            my_move=iterativedeepening(n=root, a=-INF, b=INF, DEPTH_stop=5,time_limit=5,my_color=my_color,opp_color=opp_color)
            print('n_nodes=',n_nodes,'\n cutoff=',cutoff)

        elif algor=="mcts":
            my_move=monte_carlo_tree_search(root,cp=1,N=1000)

        # retunred variable "my_move" is a node and contains info about computer's next step
        Game.place(coordinates=my_move.location,color=my_move.color)
        Game.print()

        if Game.is_game_over():
            break
        else:
            # read human's next move
            x,y = xy_read(Game)
            Game.place(coordinates=(x,y),color=opp_color)
            Game.print()
            root=Tree(Game,color=opp_color,parent=None,coordinate=(x,y),my_color=my_color)

    if Game.check_win(opp_color):
        print('Game over! You win :-)')
    else:
        print('Game over! You lose :-(')
#--------------------------------------------------------------------------------
#-------------------------------alphabeta----------------------------------------
def alphabeta(n, a, b, d, method,depth,my_color,opp_color):
    # global variables to store the things throughout the recursion
    global best_move,n_nodes,cutoff,flag

    # firstly, add children to the root node
    n.add_child_ab()

    # d<=0: if reach the last level of the tree, return evaluation value of
    # this path(from root node to leaf) based on method
    if d <= 0:
        if method=='random':
            return eval_random(n)
        elif method=='dijkstra':
            return eval_dijk(n,my_color,opp_color)

    # if not reach the end of the tree, based on n.color to decide run which
    # turn(max:computer aims to maximize g/min:opponent aims to minimize g), recursionly
    # call alphabeta itself with input n=children to get the evaluation g-value,
    # update g, pass g upwards, simultaneously update (a,b) as kind of boundary condition
    # in order to do cutoff.
    elif n.color == my_color:
        g = -np.copy(INF)
        for c in n.children:
            n_nodes += 1            # count how many nodes been searched
            temp = np.copy(alphabeta(c, a, b, d - 1, method,depth,my_color,opp_color).gvalue)
            g = max(g, temp)
            n.gvalue = np.copy(g)   # store g-value in n(type(n)=tree)

            # Update alpha: shrink boundary, alpha is to decide where we do cutoff
            a = max(a, g)
            if a >= b:
                cutoff += 1
                break
    elif n.color == opp_color:
        g = np.copy(INF)
        for c in n.children:
            n_nodes += 1            # count how many nodes been searched
            temp = np.copy(alphabeta(c, a, b, d - 1, method,depth,my_color,opp_color).gvalue)
            g = min(g, temp)        # store g-value in n(type(n)=tree)
            n.gvalue = np.copy(g)

            # Update beta: shrink boundary, alpha is to decide where we do cutoff
            b = min(b, g)
            if a >= b:
                cutoff += 1
                break

    # d==depth: if d back to the root, return the best_move(type(best_move)=node)
    if d == depth:
        return best_move
    # d==depth-1: second level of the tree contains the possible placement decisions
    # that computer could do, since we choose computer_color(=my_color) being max turn,
    # we pick the largest g value among nodes in this level, store in global variable best_move.
    elif d == depth-1:
        if flag == 0:
            flag = 1
            #print(n.gvalue)
            best_move = copy.deepcopy(n)
        if best_move.gvalue<n.gvalue:
            best_move = copy.deepcopy(n)
        return n
    else:
        return n
#--------------------------------------------------------------------------------
def eval_random(n):
    """return random number"""
    random_number = np.random.randint(1, 100)
    n.gvalue = random_number
    return n
#--------------------------------------------------------------------------------
def eval_dijk(n,my_color,opp_color):
    """a greedy search algorithm, aims to find the shortest path between nodes
    in a graph. """

    # Graph (same for one board): store the cost of that point if we want to go through it,
    # like possibility distribution function;

    # distance (depend on start point, have board.size numbers for one board):
    # store the accumulated cost starting from one of the edge points
    # to all other points in the board, like accumulative distribution function.

    # Since in this evaluation function, we recursively call a "recrusion" function,
    # the Graph, distance array should be keep updated, so to be global.
    global Graph,distance

    size=n.state.size

    # generate cost Graph, if the position(x,y) is empty cost: 1,
    # my color cost: 0, opponent cost: 99.
    Graph = np.ones((size,size))*99
    for i in range(size):
        for j in range(size):
            if n.state.get_color((i,j))==my_color:
                Graph[j,i] = 0
            if n.state.is_empty((i,j)):
                Graph[j,i] = 1

    # the connection direction of bridge, BLUE: left to right; RIGHT: top to bottom
    # also the start points are a row from the top or a column from the left
    # depending on computer color(my_color)
    for i in range(size):
        if my_color==2:
            x = 0
            y = i
        else:
            x = i
            y = 0

        # to store the output sum(distance) of one row/column start points
        # the reason why chooose sum(distance), not np.where(distance==min(distance))
        # is the board size is small(=3), the possible return results are [0,1,2,3,99(ifinity)]
        # the choice is little, moreover with search depth=3/4, we simulated
        # a lot of board (large N) with the same guess movement of the second row,
        # so this thing is very likely to happen that the returned g value in second
        # level of the tree will have distance="0" this option,
        # then the best_move will always be the first added child node leading to a regular pattern
        # which is no more what "dijkstra"'s goal.
        dist_list =[]

        # initially set every elements in distance array to a large number
        # set distance[start point] in line with Graph array, because accumulated
        # distance of the start point is just the one step cost.
        distance = np.ones((size,size))*INF
        distance[(x,y)] = np.copy(Graph[(x,y)])
        neighbors = n.state.get_neighbors((y,x))

        # a array to store if is updated, it is sligtly different from a visited array
        # when the distance value in that position can be updated(new path has better value),
        # set it to True, and onlyif index[position]=True, we would continue getting its
        # neighbor by calling recursion function. More details can be found in our report.
        index = np.zeros((size,size),dtype=bool)
        # start calculate the accumulative distance of center position's neighbor
        # by adding the center distance and neighbor cost in Graph array.
        for neighbor in neighbors:
            temp = distance[(x,y)] + Graph[neighbor[1],neighbor[0]]
            if temp<distance[neighbor[1],neighbor[0]]:
                distance[neighbor[1],neighbor[0]] = copy.deepcopy(temp)
                index[neighbor[1],neighbor[0]] = True

        # after traveling all the neighbors, and update the distance of them,
        # set current neighbor being center, find their neighbor as well by calling recursion function
        for neighbor in neighbors:
            if (not n.state.border(my_color, neighbor)) and index[neighbor[1],neighbor[0]]:
                recursion(neighbor,n,my_color)

        # record (0,i) or (i,0) disatnce result
        dist_list.append(np.sum(distance))

    # since in alphabeta, we want to maximize computer g value from -INF,
    # the return evaluation number should meet: the better, the smaller criterion.
    # thus we times -1 on it.
    n.gvalue = -min(dist_list)


    # In the board game, when we want to make a chioce, we have to consider opponent's advantages
    # as well if we decide to make a move, thus we calculate the opponent's distance array with same routine
    # but this time with a plus sign, so you could see in the last row, the n.gvalue is added
    # with opponent's sum(distance).

    Graph = np.ones((size,size))*99
    for i in range(size):
        for j in range(size):
            if n.state.get_color((i,j))==opp_color:
                Graph[j,i] = 0
            if n.state.is_empty((i,j)):
                Graph[j,i] = 1

    for i in range(size):
        if opp_color==1:
            x = 0
            y = i
        else:
            x = i
            y = 0

        dist_list =[]
        distance = np.ones((size,size))*INF
        distance[(x,y)] = np.copy(Graph[(x,y)])
        neighbors = n.state.get_neighbors((y,x))

        index = np.zeros((size,size),dtype=bool)

        for neighbor in neighbors:
            temp = distance[(x,y)] + Graph[neighbor[1],neighbor[0]]
            if temp<distance[neighbor[1],neighbor[0]]:
                distance[neighbor[1],neighbor[0]] = copy.deepcopy(temp)
                index[neighbor[1],neighbor[0]] = True


        for neighbor in neighbors:
            if (not n.state.border(opp_color, neighbor)) and index[neighbor[1],neighbor[0]]:
                recursion(neighbor,n,opp_color)

        dist_list.append(np.sum(distance))

    n.gvalue += min(dist_list)

    return n
#--------------------------------------------------------------------------------------
def recursion(center,n,color_):
    """a function used inside eval_dijk function"""
    size=n.state.size
    index = np.zeros((size,size),dtype=bool)

    neighbors = n.state.get_neighbors(center)
    for neighbor in neighbors:
        temp = distance[center[1],center[0]] + Graph[neighbor[1],neighbor[0]]
        if temp<distance[neighbor[1],neighbor[0]]:
            distance[neighbor[1],neighbor[0]] = copy.deepcopy(temp)
            index[neighbor[1],neighbor[0]] = True

    for neighbor in neighbors:
        if (not n.state.border(color_, neighbor)) and index[neighbor[1],neighbor[0]]:
            recursion(neighbor,n,color_)

#--------------------------------------------------------------------------------
#--------------------Iterative deepening and transposition tables----------------
def time_is_up(t0,time_limit):
    """return if the time is up"""
    if time.time()-t0 > time_limit:
        print('Time is up!\n use currently best move.')
        return True
    else:
        return False
#--------------------------------------------------------------------------------
def iterativedeepening(n, a, b, DEPTH_stop,time_limit,my_color,opp_color):
    global flag, best_move,transpositiontable
    t0 = time.time()
    d = 1
    transpositiontable = dict()
    # deepening
    while (not time_is_up(t0,time_limit)) and (d<=DEPTH_stop):
        best_move = ()
        flag = 0
        f = ttalphabeta(n, a, b, d, d,my_color=my_color,opp_color=opp_color)
        d = d + 1
    return f
#--------------------------------------------------------------------------------
def store(n, g, d, bm):
    """store the transposition table with dictionary type, take node as 'key',
    gvalue, delth, bestmove 'value' """
    global transpositiontable
    if n in transpositiontable.keys():
        if d>transpositiontable[n][1]:
            transpositiontable[n] = [g,d,bm]
    else:
        transpositiontable[n] = [g,d,bm]
#--------------------------------------------------------------------------------
def lookup(n, d):
    """with input (n,d), lookup if there is the same node under same depth
    has been searched before"""
    hit = False
    g = 0
    ttbm = []
    if n in transpositiontable.keys():
        g = transpositiontable[n][0]
        ttbm = transpositiontable[n][2]
        if d==transpositiontable[n][1]:
            hit = True
            # if find the totally same node at that depth
    return hit,g,ttbm
#--------------------------------------------------------------------------------
def ttalphabeta(n, a, b, d, DEPTH,my_color,opp_color):
    global n_nodes,cutoff,best_move,flag

    hit, g, ttbm = lookup(n,d)
    # if totally find the same node under same depth, return preivous result dictectly;
    # otherwise we could still use the best_move variable info to make good ordering
    if hit:
        return g

    n.add_child_ab()
    if d <= 0:
        bm = ()
        g = eval_dijk(n,my_color=my_color,opp_color=opp_color)

    elif n.color == my_color:
        g = -np.copy(INF)

        children_copy = copy.deepcopy(n.children)

        # let node ttbm be in the first place to make good ordering.
        if ttbm!=[]:
            children_copy.insert(0,ttbm)
        bm =()

        for c in children_copy:
            n_nodes += 1
            temp = np.copy(ttalphabeta(c, a, b, d - 1,DEPTH, my_color=my_color, opp_color=opp_color).gvalue)
            if temp>g:
                bm = copy.deepcopy(c)
            g = max(g, temp)
            n.gvalue = np.copy(g)
            c.gvalue = np.copy(g)
            a = max(a, g)  # Update alpha
            if a >= b:  # cutoff
                cutoff += 1
                break

    elif n.color == opp_color:
        g = np.copy(INF)

        children_copy = copy.deepcopy(n.children)
        bm = ()
        if ttbm!=[]:
            children_copy.insert(0,ttbm)

        for c in children_copy:
            n_nodes += 1
            temp = np.copy(ttalphabeta(c, a, b, d - 1,DEPTH,my_color=my_color,opp_color=opp_color).gvalue)
            if temp<g:
                bm = copy.deepcopy(c)
            g = min(g, temp)
            n.gvalue = np.copy(g)
            b = min(b, g)  # Update alpha
            if a >= b:  # cutoff
                cutoff += 1
                break
    if bm!=():
        store(n, g, d, bm)

    if d == DEPTH:
        return best_move
    elif d == DEPTH-1:
        if flag == 0:
            flag = 1
            best_move = copy.deepcopy(n)
        elif best_move.gvalue<n.gvalue:
            best_move = copy.deepcopy(n)
        return best_move
    else:
        return n

#--------------------------------------------------------------------------------
#--------------------Monte Varlo Tree search-------------------------------------
def monte_carlo_tree_search(root,cp,N):
    #duration is in minutes
    iteration=0
    while iteration<N:
        #select an unexpanded leaf
        leaf=select(root,cp)
        simulation_result=rollout(leaf,root.color)
        backpropagate(leaf,simulation_result)
        iteration+=1
    return best_child(root,0)
#--------------------------------------------------------------------------------
def select(node,cp):
    while node.is_fully_expanded():
        #terminal node also returns true
        if node.is_terminal_node():
            return node
        else:
            node=best_child(node,cp)
    #find the expanded leaf
    return node.add_child()
#--------------------------------------------------------------------------------
def rollout(node,color):
    simulate_board=copy.deepcopy(node.state)
    possible_move=simulate_board.getMoveList()
    current_color=simulate_board.get_opposite_color(node.color)
    while not simulate_board.is_game_over():
        pick=np.random.randint(len(possible_move))
        simulate_board.place(possible_move.pop(pick),current_color)
        current_color=simulate_board.get_opposite_color(current_color)
    return simulate_board.check_win(node.state.get_opposite_color(color))
#--------------------------------------------------------------------------------
def backpropagate(node,result):
    node.visit+=1
    if result==True:
        node.win+=1
    if node.parent==None:
        return
    backpropagate(node.parent,result)
#--------------------------------------------------------------------------------
def best_child(node, cp=1):
    choice_weights=[]
    for c in node.children:
        choice_weights.append(c.win/c.visit+cp*np.sqrt(np.log(node.visit)/c.visit))
    return node.children[np.argmax(choice_weights)]

#--------------------------------------------------------------------------------
#--------------functions used in agent ns agent experiments----------------------
def alphabeta_randomVSdijkstra(method1='random',method2='dijkstra',depth1=3,depth2=3,size=3,print_all=False,first=True):
    #alphabeta: method1 and method2 can choose 'random' or 'dijkstra'
    #first==True: method1 moves first
    #first determins which moves first
    global flag, user_color,unuse_color,INF, n_nodes, cutoff
    Game=HexBoard(size)
    #firt move color=A1_color: blue
    A1_color=1
    #second move color=A1_color: red
    A2_color=2
    root=Tree(Game,color=A2_color,parent=None,coordinate=None,my_color=A1_color)
    if print_all:
        if first==True:
            print('First move (blue) is '+method1)
            print('Second move (red) is '+method2)
        else:
            print('First move (blue) is '+method2)
            print('Second move (red) is '+method1)

    while not Game.is_game_over():
        INF = 99999
        flag = 0
        n_nodes = 0
        cutoff = 0
        if first==True:
            #method1 moves first
           A1_move=alphabeta(n=root, a=-INF, b=INF, d=depth1, method=method1,depth=depth1,my_color=A1_color,opp_color=A2_color)
        else:
            #method2 moves first
           A1_move=alphabeta(n=root, a=-INF, b=INF, d=depth2, method=method2,depth=depth2,my_color=A1_color,opp_color=A2_color)
        Game.place(coordinates=A1_move.location,color=A1_color)
        if Game.is_game_over():
            if print_all:
                Game.print()
            if first==True:
                #if method1 win, return True
                return Game.check_win(A1_color)
            else:
                #if method1 win, return True
                return Game.check_win(A2_color)
        else:
            root=Tree(Game,color=A1_color,parent=None,coordinate=A1_move.location,my_color=A2_color)
            n_nodes = 0
            cutoff = 0
            flag = 0
            if first==True:
               A2_move=alphabeta(n=root, a=-INF, b=INF, d=depth2, method=method2,depth=depth2,my_color=A2_color,opp_color=A1_color)
            else:
               A2_move=alphabeta(n=root, a=-INF, b=INF, d=depth1, method=method1,depth=depth1,my_color=A2_color,opp_color=A1_color)
            Game.place(coordinates=A2_move.location,color=A2_move.color)
            root=Tree(Game,color=A2_color,parent=None,coordinate=A2_move.location,my_color=A1_color)
    if print_all:
        Game.print()


    if first==True:
        #if method1 win or not
        return Game.check_win(A1_color)
    else:
        return Game.check_win(A2_color)

#--------------------------------------------------------------------------------------------------------------
def idtt_alphabeta(method='random',idtt_depth=3,depth2=3,size=3,print_all=False,first=True,time_limit=5):
    #idtt against alphabeta (random or dijkstra)
    #idtt_depth is idtt's depth; depth2 is method's deptth
    #first determins which moves first
    global flag,INF, n_nodes, cutoff
    Game=HexBoard(size)
    #firt move color=A1_color: blue
    A1_color=1
    #second move color=A1_color: red
    A2_color=2
    root=Tree(Game,color=A2_color,parent=None,coordinate=None,my_color=A1_color)
    if print_all:
        if first==True:
            print('First move (blue) is idtt')
            print('Second move (red) is '+method)
        else:
            print('First move (blue) is '+method)
            print('Second move (red) is idtt')

    while not Game.is_game_over():
        INF = 99999
        flag = 0
        n_nodes = 0
        cutoff = 0
        if first==True:
            #idtt moves first
           A1_move=iterativedeepening(n=root, a=-INF, b=INF, DEPTH_stop=idtt_depth,time_limit=time_limit,my_color=A1_color,opp_color=A2_color)
        else:
            #method2 moves first
           A1_move=alphabeta(n=root, a=-INF, b=INF, d=depth2, method=method,depth=depth2,my_color=A1_color,opp_color=A2_color)
        Game.place(coordinates=A1_move.location,color=A1_color)
        if Game.is_game_over():
            if print_all:
                Game.print()
            if first==True:
                #if idtt win, return True
                return Game.check_win(A1_color)
            else:
                #if idtt win, return True
                return Game.check_win(A2_color)
        else:
            root=Tree(Game,color=A1_color,parent=None,coordinate=A1_move.location,my_color=A2_color)
            n_nodes = 0
            cutoff = 0
            flag = 0
            if first==True:
               A2_move=alphabeta(n=root, a=-INF, b=INF, d=depth2, method=method,depth=depth2,my_color=A2_color,opp_color=A1_color)
            else:
               A2_move=iterativedeepening(n=root, a=-INF, b=INF, DEPTH_stop=idtt_depth,time_limit=time_limit,my_color=A2_color,opp_color=A1_color)
            Game.place(coordinates=A2_move.location,color=A2_move.color)
            root=Tree(Game,color=A2_color,parent=None,coordinate=A2_move.location,my_color=A1_color)
    if print_all:
        Game.print()


    if first==True:
        #if method1 win or not
        return Game.check_win(A1_color)
    else:
        return Game.check_win(A2_color)

#-----------------------------------------------------------------------------------------------------------
def mcts_alphabeta(cp=1, N=500,method='dijkstra',depth=3,size=3,print_all=False,first=True,time_limit=5):
    #mcts plays with method
    #method='dijkstra' or 'idtt'
    global flag, user_color,unuse_color,INF, n_nodes, cutoff
    Game=HexBoard(size)
    A1_color=1
    A2_color=2
    root=Tree(Game,color=A2_color,parent=None,coordinate=None,my_color=A1_color)
    if print_all:
        if first==True:
            print('First move (blue) is mcts')
            print('Second move (red) is '+method)
        else:
            print('First move (blue) is '+method)
            print('Second move (red) is mcts')

    while not Game.is_game_over():
        INF = 99999
        flag = 0
        n_nodes = 0
        cutoff = 0
        if first==True:
            #mcts moves first
           A1_move=monte_carlo_tree_search(root,cp=cp,N=N)
        else:
            #method moves first
            if method=='dijkstra':
               A1_move=alphabeta(n=root, a=-INF, b=INF, d=depth, method=method,depth=depth,my_color=A1_color,opp_color=A2_color)
            elif method=='idtt':
               A1_move=iterativedeepening(n=root, a=-INF, b=INF, DEPTH_stop=depth,time_limit=time_limit,my_color=A1_color,opp_color=A2_color)
        Game.place(coordinates=A1_move.location,color=A1_color)
        if Game.is_game_over():
            if print_all:
                Game.print()
            if first==True:
                #if method1 win, return True
                return Game.check_win(A1_color)
            else:
                #if method1 win, return True
                return Game.check_win(A2_color)
        else:
            root=Tree(Game,color=A1_color,parent=None,coordinate=A1_move.location,my_color=A2_color)
            flag = 0
            n_nodes = 0
            cutoff = 0
            if first==True:
               if method=='idtt':
                  A2_move=iterativedeepening(n=root, a=-INF, b=INF, DEPTH_stop=depth,time_limit=time_limit,my_color=A2_color,opp_color=A1_color)
               elif method=='dijkstra':
                  A2_move=alphabeta(n=root, a=-INF, b=INF, d=depth, method=method,depth=depth,my_color=A2_color,opp_color=A1_color)
            else:
               A2_move=monte_carlo_tree_search(root,cp=cp,N=N)
            Game.place(coordinates=A2_move.location,color=A2_move.color)
            root=Tree(Game,color=A2_color,parent=None,coordinate=A2_move.location,my_color=A1_color)
    if print_all:
        Game.print()

    if first==True:
        #if method1 win or not
        return Game.check_win(A1_color)
    else:
        return Game.check_win(A2_color)

#-------------------------------------------------------------------------------------------------------------
def mcts_mcts(cp1, N1,cp2, N2, size,print_all=False,first=True):
    #mcts plays with mcts
    Game=HexBoard(size)
    A1_color=1
    A2_color=2
    root=Tree(Game,color=A2_color,parent=None,coordinate=None)
    if print_all:
        if first==True:
            print('First move (blue) is mcts')
            print('Second move (red) is mcts')
        else:
            print('First move (blue) is mcts')
            print('Second move (red) is mcts')

    while not Game.is_game_over():
        if first==True:
            #mcts moves first
           A1_move=monte_carlo_tree_search(root,cp=cp1,N=N1)
        else:
           A1_move=monte_carlo_tree_search(root,cp=cp2,N=N2)
        Game.place(coordinates=A1_move.location,color=A1_color)
        if Game.is_game_over():
            if print_all:
                Game.print()
            if first==True:
                #if method1 win, return True
                return Game.check_win(A1_color)
            else:
                #if method1 win, return True
                return Game.check_win(A2_color)
        else:
            root=Tree(Game,color=A1_color,parent=None,coordinate=A1_move.location)
            if first==True:
                  A2_move=monte_carlo_tree_search(root,cp=cp2,N=N2)
            else:
                  A2_move=monte_carlo_tree_search(root,cp=cp1,N=N1)
            Game.place(coordinates=A2_move.location,color=A2_move.color)
            root=Tree(Game,color=A2_color,parent=None,coordinate=A2_move.location)
    if print_all:
        Game.print()

    if first==True:
        #if method1 win or not
        return Game.check_win(A1_color)
    else:
        return Game.check_win(A2_color)
