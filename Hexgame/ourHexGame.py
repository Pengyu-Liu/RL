from hex_skeleton import HexBoard
import random as rd
import numpy as np
import pandas as pd
import copy
from trueskill import Rating, quality_1vs1, rate_1vs1

SIZE = 4
EMPTY = HexBoard.EMPTY
INF = 99
agent1 = HexBoard.BLUE
agent2 = HexBoard.RED

df_alphabeta = pd.DataFrame(columns=['d', 'g', 'a', 'b'])  # store d,g,a,b values
df_movelist = pd.DataFrame(columns=['x', 'y'])  # store movelist of player and AI


def alphabeta(board, d, a, b, method, agent1, agent2, type_max=True):
    global df_alphabeta, df_movelist
    if d <= 0:
        return eval(board, method)
    elif type_max == True:
        g = -INF
        for i in range(SIZE):
            for j in range(SIZE):

                if board.is_empty((i, j)):
                    board.place((i, j), agent1)

                    g_star = g
                    g = max(g, alphabeta(board, d - 1, a, b, method, agent1, agent2, type_max=False))
                    a = max(a, g)  # Update alpha.

                    df_alphabeta = df_alphabeta.append({'d': d, 'g': g, 'a': a, 'b': b}, ignore_index=True)

                    if g > g_star:
                        best_move = (i, j)
                    board.clear((i, j))

                    if a >= b:  # g>=b
                        break

    elif type_max == False:
        g = INF
        for i in range(SIZE):
            for j in range(SIZE):

                if board.is_empty((i, j)):
                    board.place((i, j), agent2)

                    g = min(g, alphabeta(board, d - 1, a, b, agent1, agent2, method, type_max=True))
                    b = min(b, g)  # Update beta

                    df_alphabeta = df_alphabeta.append({'d': d, 'g': g, 'a': a, 'b': b}, ignore_index=True)

                    board.clear((i, j))

                    if a >= b:  # a>=g
                        break
    if d == DEPTH:
        df_movelist = df_movelist.append({'x': best_move[0], 'y': best_move[1]}, ignore_index=True)
    return g



def eval(board, method):
    if method == 'random':
        random_number = rd.randint(1, 9)
        return random_number
    else:
        a1_heur_val = dijkstra_sp(board, player=agent1)
        a2_heur_val = dijkstra_sp(board, player=agent2)
        return a2_heur_val - a1_heur_val


def dijkstra_sp(board, player):
    initial = []

    if player == agent1:
        for i in range(SIZE):
            initial.append((0, i))
    if player == agent2:
        for i in range(SIZE):
            initial.append((i, 0))

    # Make a distance graph.
    distance_graph = np.zeros((SIZE, SIZE))
    distance_graph.fill(INF)

    for i in range(len(initial)):
        now = initial[i]
        visited = []
        distance_graph[now[1], now[0]] = 0

        if player == agent1:
            a1_update_d(board, now, distance_graph, visited)
        if player == agent2:
            a2_update_d(board, now, distance_graph, visited)

    if player == agent1:
        return min(distance_graph[:, -1])
    if player == agent2:
        return min(distance_graph[-1])


def a1_update_d(board, now, distance_graph, visited):
    border_reached = False

    if board.border(agent1, now) == True:
        border_reached = True

    cur_distance = distance_graph[now[1], now[0]]
    shortest = INF
    next = []

    neighbors = board.get_neighbors(now)

    for i in range(len(neighbors)):
        if neighbors[i] not in visited and board.is_color(neighbors[i], agent2) == False:
            next_distance = cur_distance + 1

            if board.is_color(neighbors[i], agent1) == True and cur_distance < distance_graph[
                neighbors[i][1], neighbors[i][0]]:
                distance_graph[neighbors[i][1], neighbors[i][0]] = cur_distance
                next = neighbors[i]
                shortest = cur_distance
            elif next_distance < distance_graph[neighbors[i][1], neighbors[i][0]]:
                distance_graph[neighbors[i][1], neighbors[i][0]] = next_distance

            if next_distance < shortest:
                next = neighbors[i]
                shortest = next_distance

    visited.append(now)

    if (now[0] + 1, now[1]) in neighbors:
        for i in range(len(neighbors)):
            if board.is_color(neighbors[i], agent1) == True:
                break
            else:
                next = (now[0] + 1, now[1])

    if len(next) != 0 and border_reached == False:
        a1_update_d(board, next, distance_graph, visited)


def a2_update_d(board, now, distance_graph, visited):
    border_reached = False

    if board.border(agent2, now) == True:
        border_reached = True

    cur_distance = distance_graph[now[1], now[0]]
    shortest = INF
    next = []

    neighbors = board.get_neighbors(now)

    for i in range(len(neighbors)):
        if neighbors[i] not in visited and board.is_color(neighbors[i], agent1) == False:
            next_distance = cur_distance + 1

            if board.is_color(neighbors[i], agent2) == True and cur_distance < distance_graph[
                neighbors[i][1], neighbors[i][0]]:
                distance_graph[neighbors[i][1], neighbors[i][0]] = cur_distance
                next = neighbors[i]
                shortest = cur_distance
            elif next_distance < distance_graph[neighbors[i][1], neighbors[i][0]]:
                distance_graph[neighbors[i][1], neighbors[i][0]] = next_distance

            if next_distance < shortest:
                next = neighbors[i]
                shortest = next_distance

    visited.append(now)

    if (now[0], now[1] + 1) in neighbors:
        for i in range(len(neighbors)):
            if board.is_color(neighbors[i], agent2) == True:
                break
            else:
                next = (now[0], now[1] + 1)

    if len(next) != 0 and border_reached == False:
        a2_update_d(board, next, distance_graph, visited)


def ai_make_move(board):
    move = df_movelist.to_numpy()
    x = move[-1, 0]
    y = move[-1, 1]

    make_move = (int(x), int(y))
    board.place(make_move, agent1)
    board.print()


def pl_make_move(board):
    global df_movelist
    print('Next move.')
    x = int(ord(input(' x(a,b,c...): ')) - 97)
    y = int(input(' y(0,1,2...): '))

    while not board.is_empty((x, y)):
        print('Invaild place, input again!!!')
        x = int(ord(input(' x(a,b,c...): ')) - 97)
        y = int(input(' y(0,1,2...): '))

    df_movelist = df_movelist.append({'x': x, 'y': y}, ignore_index=True)

    board.place((x, y), agent2)
    board.print()


# Play the game.
def play_game(board, method):
    for i in range(SIZE * SIZE):
        board_copy = copy.deepcopy(board)

        eval_val = alphabeta(board_copy, DEPTH, -INF, INF, agent1, agent2, method)
        df_alphabeta.to_csv('alphabeta.txt', index=False, mode='a')
        ai_make_move(board)
        if board.check_win(agent1):
            print('A1 WINS')
            break

        pl_make_move(board)
        if board.check_win(agent2):
            print('A2 WINS')
            break


DEPTH = 3
board = HexBoard(SIZE)

play_game(board, 'Dijkstra')
