# Sudoku Solver

## 使用模块

本项目根据三种方法分为

BruteForce / BTCSP / FCMRV	三个模块

```python
import os
from BTCSP import BT_CSP
from BruteForce import brute_force
from FCMRV import FC_MRV
```

## 可使用函数

分别可调用

solve( sudoku )函数 -> 得到第一个答案 并 输出该答案 和 搜索/fill的节点数 和 运行时间

multisolve (sudoku)函数 -> 输出所有答案个数 和 搜索/fill的节点数 和 运行时间

## main.py的主函数

```python
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
```