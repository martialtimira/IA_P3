#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import csv

import chess
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False
        self.stateDict = {}
        self.actionDict = {}
        self.qTable = []
        self.rewardTable = []

    def getCurrentState(self):

        return self.chess.boardSim.currentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    def isCheckMate(self, mystate):
        checkMateStates = [[[0, 0, 2], [2, 4, 6]], [[0, 1, 2], [2, 4, 6]], [[0, 2, 2], [2, 4, 6]],
                           [[0, 6, 2], [2, 4, 6]], [[0, 7, 2], [2, 4, 6]]];
        perm_state = list(permutations(mystate))

        for j in range(len(perm_state)):
            for k in range(len(checkMateStates)):

                if self.isSameState(list(perm_state[j]), checkMateStates[k]):
                    self.checkMate = True

            return self.checkMate

    def getMoveFromStates(self, currentState, nextState):
        """
        Returns the "start" and "to" points of a move from its 2 states
        Args:
            currentState: Current State of the board
            nextState: State of the Board after the move

        Returns: Starting coordinates, To coordinates, piece ID

        """
        start = None
        to = None
        piece = None

        for element in currentState:                    #compare each element of both states, to find the one in the current state that isn't
            if element not in nextState:                #on the next state, and define that one as the starting point, also define which piece it is
                start = (element[0], element[1])
                piece = element[2]
        for element in nextState:                       #repeat, but instead find the one in nextState that isn't in currentState, and
            if element not in currentState:             #define that one as the "to" point.
                to = (element[0], element[1])


        return start, to, piece

    def getIndexFromAction(self, start, to, piece):
        '''
        Returns an index for the action recieved
        Args:
            start:
            to:
            piece:

        Returns:

        '''
        diffY = start[0] - to[0]
        diffX = start[1] - to[1]
        if piece == 6:          ##King actions
            if diffY == 1 and diffX == 0:
                return 0
            if diffY == 1 and diffX == 1:
                return 1
            if diffY == 1 and diffX == -1:
                return 2
            if diffY == 0 and diffX == 1:
                return 3
            if diffY == 0 and diffX == -1:
                return 4
            if diffY == -1 and diffX == 0:
                return 5
            if diffY == -1 and diffX == 1:
                return 6
            if diffY == -1 and diffX == -1:
                return 7

        if piece == 2:          ##Tower Actions
            ##Tower up actions (up 1 to up7)
            if diffY == 1 and diffX == 0:
                return 8
            if diffY == 2 and diffX == 0:
                return 9
            if diffY == 3 and diffX == 0:
                return 10
            if diffY == 4 and diffX == 0:
                return 11
            if diffY == 5 and diffX == 0:
                return 12
            if diffY == 6 and diffX == 0:
                return 13
            if diffY == 7 and diffX == 0:
                return 14
            ##Tower down actions (down 1 to down7)
            if diffY == -1 and diffX == 0:
                return 15
            if diffY == -2 and diffX == 0:
                return 16
            if diffY == -3 and diffX == 0:
                return 17
            if diffY == -4 and diffX == 0:
                return 18
            if diffY == -5 and diffX == 0:
                return 19
            if diffY == -6 and diffX == 0:
                return 20
            if diffY == -7 and diffX == 0:
                return 21
            ##Tower Left actions (left 1 to left7)
            if diffY == 0 and diffX == 1:
                return 22
            if diffY == 0 and diffX == 2:
                return 23
            if diffY == 0 and diffX == 3:
                return 24
            if diffY == 0 and diffX == 4:
                return 25
            if diffY == 0 and diffX == 5:
                return 26
            if diffY == 0 and diffX == 6:
                return 27
            if diffY == 0 and diffX == 7:
                return 28
            ##Tower Left actions (left 1 to left7)
            if diffY == 0 and diffX == -1:
                return 29
            if diffY == 0 and diffX == -2:
                return 30
            if diffY == 0 and diffX == -3:
                return 31
            if diffY == 0 and diffX == -4:
                return 32
            if diffY == 0 and diffX == -5:
                return 33
            if diffY == 0 and diffX == -6:
                return 34
            if diffY == 0 and diffX == -7:
                return 35
        return None

def init_state_dict():
    # El csv nomes conte les poscicions de les peces, no el tipus de peça.
    # Les dues primeres columnes son la torra i les dues segones el rei

    with open('Estats_IA_P3.csv', mode='r') as file:
        csvFile = csv.reader(file)

        states = []  # Matriu amb els estats

        for line in csvFile:
            states.append([int(x) for x in line])

        estats = {}

        for s in range(len(states)):
            estats.update({repr([[states[s][0], states[s][1], 2], [states[s][2], states[s][3], 6]]): s})
    return estats

def init_tables(stateDict):
    qTable = np.zeros((len(stateDict), 36))
    rewardTable = np.full((len(stateDict), 36), -1)

    #Set reward of all the tower actions that end in CheckMate to 100
    rook_check_mate_columns = [0, 1, 2, 6, 7]
    for column in rook_check_mate_columns:
        for row in range(7, 0, -1):
            currentState = [[row, column, 2], [2, 4, 6]]
            checkMateState = [[0, column, 2], [2, 4, 6]]
            start, to, piece = aichess.getMoveFromStates(currentState, checkMateState)
            actionIndex = aichess.getIndexFromAction(start, to, piece)
            stateIndex = stateDict[repr(currentState)]
            rewardTable[stateIndex][actionIndex] = 100

        #set reward of all the king actions that lead to CheckMate to 100
        currentState_1 = [[0, column, 2], [2, 3, 6]]
        currentState_2 = [[0, column, 2], [2, 5, 6]]
        currentState_3 = [[0, column, 2], [3, 3, 6]]
        currentState_4 = [[0, column, 2], [3, 4, 6]]
        currentState_5 = [[0, column, 2], [3, 5, 6]]
        checkMateStateK = [[0, column, 2], [2, 4, 6]]

        king_states = [currentState_1, currentState_2, currentState_3, currentState_4, currentState_5]

        for state in king_states:
            start, to, piece = aichess.getMoveFromStates(state, checkMateStateK)
            actionIndex = aichess.getIndexFromAction(start, to, piece)
            stateIndex = stateDict[repr(state)]
            rewardTable[stateIndex][actionIndex] = 100

    return qTable, rewardTable



def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    # TA[0][0] = 2
    # TA[2][4] = 6
    # # black pieces
    # TA[0][4] = 12

    TA[0][0] = 2
    #TA[7][4] = 6
    TA[3][4] = 6
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State", currentState)

    # it uses board to get them... careful 
    aichess.getListNextStatesW(currentState)
    #   aichess.getListNextStatesW([[7,4,2],[7,4,6]])
    print("list next states ", aichess.listNextStates)

    # starting from current state find the end state (check mate) - recursive function
    # aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
    depth = 0

    # MovesToMake = ['1e','2e','2e','3e','3e','4d','4d','3c']

    # for k in range(int(len(MovesToMake)/2)):

    #     print("k: ",k)

    #     print("start: ",MovesToMake[2*k])
    #     print("to: ",MovesToMake[2*k+1])

    #     start = translate(MovesToMake[2*k])
    #     to = translate(MovesToMake[2*k+1])

    #     print("start: ",start)
    #     print("to: ",to)

    #     aichess.chess.moveSim(start, to)

    # aichess.chess.boardSim.print_board()
    ##Initialize state dictionary, qTable and Reward Table, and assign them to aichess
    state_dict = init_state_dict()
    qTable, rewardTable =  init_tables(state_dict)

    ##From here we can reinitialize Aichess and keep the data of QTable and rewardTable
    aichess.qTable = qTable
    aichess.rewardTable = rewardTable
    aichess.stateDict = state_dict
    print("Allstates: ", len(state_dict))
    start, to, piece = aichess.getMoveFromStates(aichess.getCurrentState(), [[0, 0,2], [2,4,6]])
    actionIndex = aichess.getIndexFromAction(start, to, piece)
    currentState = aichess.getCurrentState()
    currentState.sort(key=lambda x: x[2])
    print("CS: ", currentState)
    stateIndex = aichess.stateDict[repr(currentState)]
    print("AIndex: ", actionIndex, "SIndex: ", stateIndex)
    print("Reward for those indexes: ", aichess.rewardTable[stateIndex][actionIndex])
    aichess.chess.moveSim(start, to)
    aichess.chess.boardSim.print_board()
    print("Action to make: ", start, to, piece)
    print("Index of action: ", aichess.getIndexFromAction(start, to, piece))
    print("#Move sequence...  ", aichess.pathToTarget)
    print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State...  ", aichess.chess.board.currentStateW)
    print("IsCheckMate: ", aichess.isCheckMate(aichess.chess.boardSim.currentStateW))
