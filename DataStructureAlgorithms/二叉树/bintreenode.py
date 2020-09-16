from queue import SQueue
from stack import SStack

class BinTNode():
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def is_empty(self):
        return self.data is None
    
def count_nodes(t):
    if t.is_empty():
        return 0
    else:
        return 1 + count_nodes(t.left) + count_nodes(t.right) #递归形式

def sum_nodes_data(t):
    if t.is_empty():
        return 0
    
    else:
        return t.data + sum_nodes_data(t.left) + sum_nodes_data(t.right) #递归形式

def dfs_recursive_traverse(t,proc): #dfs递归 proc是具体的结点操作 lambda表达式 先根序
    if t.is_empty():
        return 
    
    proc(t.data)
    dfs_recursive_traverse(t.left,proc) #递归方法实现DFS
    dfs_recursive_traverse(t.right,proc)

def wfs_traverse(t,proc): #wfs 先根序
    qu = SQueue()
    qu.enqueue(t)
    while not qu.is_empty():
        t = qu.dequeue()
        if t is None:
            continue
        qu.enqueue(t.left)
        qu.enqueue(t.right)
        proc(t.data)

def dfs_non_recursive_traverse(t,proc): #dfs非递归 先根序
    s = SStack()

    while t is not None or not s.is_empty():
        while t is not None:
            proc(t.data)
            s.push(t.right)
            t = t.left
        
        t = s.pop()

t = BinTNode(1, BinTNode(2), BinTNode(3))
dfs_non_recursive_traverse(t, lambda x : print(x,end=" ")) #lambda表达式

def dfs_non_recursive_traverse_postorder(t , proc): #dfs非递归 后根序
    s = SStack()

    while t is not None or not s.is_empty():
        while t is not None:
            s.push(t)
            if t.left is not None:
                t = t.left
            else:
                t = t.right
            
        t = s.pop()
        proc(t.data)
        if not s.is_empty() and s.top().left == t:
            t = s.top().right #这一步很关键
        else:
            t = None