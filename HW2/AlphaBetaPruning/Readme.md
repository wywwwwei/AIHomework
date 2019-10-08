# AlphaBetaPruning

在该文件中，游戏格局树采用

[ root , [ child1 , child2 , child3……]]

的列表形式来表示

根据伪代码

```pseudocode
Function:(move)minimax(state, turn, alpha, beta)
Input:	State of the game, Whose turn it is, Alpha, Beta
Output: the best move that can be played by the player given in the input.

if game is in terminal state then
	return static score of node
	
if turn == Maximizer then
	maxScore= -infinity            
 	for each child of node do
    	curState= move to this child
 		curScore= minimax(curState,Minimizer,alpha,beta)  
		maxScore= max(maxScore,curScore)        //gives Maximum of the values
        alpha= max(alpha,maxScore)				//Update alpha
        if beta <= alpha then
        	break
	return maxScore
else
	minScore= infinity            
 	for each child of node do
    	curState= move to this child
 		curScore= minimax(curState,Maximizer)  
		minScore= min(maxScore,curScore)        //gives Minimum of the values  
		beta= min(beta,minScore)				//Update beta
        if beta <= alpha then
        	break
	return minScore
```

编写