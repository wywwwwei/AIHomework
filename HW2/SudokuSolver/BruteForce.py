# -*- coding: UTF-8 -*-
import time

class brute_force:
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
        print("brute force:")
        for i in range(9):
            print(result[i*9:(i+1)*9])
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)

    def multiprint(self):
        print("brute force:")
        print("ResutlNum:",self.result_num)
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)

    def satisfied(self,sudoku,i):
        row_element = [sudoku[j] for j in range(81)
                        if not (i // 9 ^ j // 9) and sudoku[j]!=0]
        col_element = [sudoku[j] for j in range(81)
                        if not (i-j) % 9 and sudoku[j]!=0]
        box_element = [sudoku[j] for j in range(81)
                        if not (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3))and sudoku[j]!=0]
        return len(row_element)==len(set(row_element)) and len(col_element)==len(set(col_element)) and len(box_element)==len(set(box_element))

    def solver(self,sudoku):
        try:
            i = sudoku.index(0)
        except ValueError:
            return sudoku

        for v in range(1, 10):
            self.nodes += 1

            new_sudoku = sudoku[:i] + [v] + sudoku[i + 1:]
            if not self.satisfied(new_sudoku,i):
                continue

            r = self.solver(new_sudoku)
            if r is not None:
                return r

    def multisolver(self,sudoku):
        try:
            i = sudoku.index(0)
        except ValueError:
            self.result_num += 1
            return sudoku

        for v in range(1, 10):
            self.nodes += 1

            new_sudoku = sudoku[:i] + [v] + sudoku[i + 1:]
            if not self.satisfied(new_sudoku,i):
                continue

            r = self.multisolver(new_sudoku)