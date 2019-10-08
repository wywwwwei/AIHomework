# -*- coding: UTF-8 -*-
import time

class BT_CSP:
    def __init__(self):
        self.nodes = 0
        self.total_clock_time = 0
        self.result_num = 0

    def solve(self,sudoku):
        start_time = time.time()
        result = self.solver(sudoku)
        end_time = time.time()

        self.total_clock_time = end_time - start_time
        self.print(result)

    def multisolve(self,sudoku):
        start_time = time.time()
        result = self.multisolver(sudoku)
        end_time = time.time()

        self.total_clock_time = end_time - start_time
        self.multiprint()

    def print(self,result):
        print("BT_CSP:")
        for i in range(9):
            print(result[i*9:(i+1)*9])
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)

    def multiprint(self):
        print("BT_CSP:")
        print("ResutlNum:",self.result_num)
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)

    def solver(self,sudoku):
        try:
            i = sudoku.index(0)
        except ValueError:
            return sudoku

        c = [sudoku[j] for j in range(81)
            if not ((i - j) % 9 * (i // 9 ^ j // 9) * (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3)))]

        for v in range(1, 10):
            if v not in c:
                self.nodes += 1
                r = self.solver(sudoku[:i] + [v] + sudoku[i + 1:])
                if r is not None:
                    return r

    def multisolver(self,sudoku):
        try:
            i = sudoku.index(0)
        except ValueError:
            self.result_num += 1
            return sudoku

        c = [sudoku[j] for j in range(81)
            if not ((i - j) % 9 * (i // 9 ^ j // 9) * (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3)))]

        for v in range(1, 10):
            if v not in c:
                self.nodes += 1
                r = self.multisolver(sudoku[:i] + [v] + sudoku[i + 1:])
