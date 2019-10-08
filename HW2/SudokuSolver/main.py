# -*- coding: UTF-8 -*-

import os
from BTCSP import BT_CSP
from BruteForce import brute_force
from FCMRV import FC_MRV

def read_file(test_file):
    sudoku = []
    with open(test_file,"r",encoding="UTF-8") as sudoku_file:
        for line in sudoku_file:
            data = line.split()
            if len(data)== 0:
                break
            for i in data:
                sudoku.append(int(i))
    return sudoku

if __name__ == '__main__':
    test_file = input("test file:")
    if not os.path.exists(test_file):
        print("wrong file path")
        exit(1)
    sudoku = read_file(test_file)
    bf_solver = brute_force()
    bf_solver.multisolve(sudoku)
    btcsp_solver = BT_CSP()
    btcsp_solver.multisolve(sudoku)
    fcmrv_solver = FC_MRV()
    fcmrv_solver.multisolve(sudoku)
