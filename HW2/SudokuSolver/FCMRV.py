# -*- coding: UTF-8 -*-
import time

class FC_MRV:
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
        print("FC_MRV:")
        for i in range(9):
            print(result[i*9:(i+1)*9])
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)
    
    def multiprint(self):
        print("FC_MRV:")
        print("ResutlNum:",self.result_num)
        print("Nodes:",self.nodes)
        print("TotalClockTime:",self.total_clock_time)

    def satisfied(self,sudoku,i):
        all_element = [sudoku[j] for j in range(81)
                        if not ((i - j) % 9 * (i // 9 ^ j // 9) * (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3)))]
        check_repeat = set(all_element)
        return len(check_repeat)==len(all_element)

    def get_index(self,sudoku):
        store_blank_mrv = []
        cores_index = []
        for i in range(81):
            if sudoku[i] == 0:
                mrv_value =  10 - len(set([sudoku[j] for j in range(81)
                            if not ((i - j) % 9 * (i // 9 ^ j // 9) * (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3)))]))
                store_blank_mrv.append(mrv_value)
                cores_index.append(i)
        return cores_index[store_blank_mrv.index(min(store_blank_mrv))]

    def solver(self,sudoku):
        try:
            i = self.get_index(sudoku)
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
            i = self.get_index(sudoku)
        except ValueError:
            self.result_num += 1
            return sudoku

        c = [sudoku[j] for j in range(81)
            if not ((i - j) % 9 * (i // 9 ^ j // 9) * (i // 27 ^ j // 27 | (i % 9 // 3 ^ j % 9 // 3)))]

        for v in range(1, 10):
            if v not in c:
                self.nodes += 1
                r = self.multisolver(sudoku[:i] + [v] + sudoku[i + 1:])
                