# -*- coding: UTF-8 -*-
INF = 10000

gameTree = [None,
            [[None,
              [[None,
                [[None,
                  [[None,
                    [[None,[0,5]],
                     [None,[-3,3]]]],
                   [None,
                    [[None,[3]],
                     [None,[-3,0,2]],
                     [None,[-2,3]]]]]]]],
               [None,
                [[None,
                  [[None,
                    [[None,[5,2]],
                     [None,[5,-5]]]]]],
                 [None,
                  [[None,
                    [[None,[0,1]]]]]]]]]],
            [None,
             [[None,
               [[None,
                 [[None,
                   [[None,[5,1]],
                    [None,[-3,0]]]]]],
                [None,
                 [[None,
                   [[None,[-5,5]]]],
                  [None,
                   [None,[-3,3]]]]]]],
              [None,
               [[None,
                 [[None,
                   [[None,[2]]]]]]]]]]]]

def alpha_beta_check(tree,turn,alpha,beta):
    if type(tree) == int:
        print("LeafNode:",tree)
        return tree

    if turn:
        maxScore = -INF
        for i in (tree[1]):
            curScore = alpha_beta_check(i,False,alpha,beta)
            maxScore = max(maxScore,curScore)
            alpha = max(alpha,maxScore)
            if beta <= alpha:
                break
        return maxScore
    else:
        minScore = INF
        for i in (tree[1]):
            curScore = alpha_beta_check(i,True,alpha,beta)
            minScore = min(minScore,curScore)
            beta = min(beta,minScore)
            if beta <= alpha:
                break
        return minScore

if __name__ == '__main__':
    print("result:", alpha_beta_check(gameTree,True,-INF,INF))