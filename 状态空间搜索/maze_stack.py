class StackUnderflow(ValueError):
    pass

class SStack(): #基于顺序表技术实现的栈类

    def __init__(self): #用list对象 _elems存储栈中元素
        self._elems = [] #所有栈操作都映射到list操作
    
    def is_empty(self):
        return self._elems == []

    def depth(self):
        return len(self._elems)
        
    def top(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.top()")

        return self._elems[-1]
    
    def push(self,elem):
        self._elems.append(elem)
    
    def pop(self):

        if self._elems == []:
            raise StackUnderflow("in SStack.pop()")
        
        return self._elems.pop()

dirs = [(0,1),(1,0),(0,-1),(-1,0)]

def mark(maze,pos): #标记搜索过的位置为2
    maze[pos[0]][pos[1]] == 2

def passable(maze,pos): #判断位置pos是否可以通行
    return maze[pos[0]][pos[1]] == 0

def maze_st(maze,start,end):
    if start == end:
        print(start)
        return 
    
    st = SStack()
    mark(maze,start)
    st.push((start,0))  #入口和方向0的序对入栈

    while not st.is_empty():
        pos, nxt = st.pop() #取栈顶及其探查方向 在下面探明方向之后还要压入栈中

        for i in range(nxt,4): #依次检查未探查方向
            nextp = (pos[0] + dirs[i][0],
                        pos[1] + dirs[i][1]) #算出下一位置
            
            if nextp == end:
                print_path(end,pos,st) #print_path打印路径
                return 
            
            if passable(maze,nextp):
                st.push((pos, i + 1))
                mark(maze,nextp)
                st.push((nextp,0))
                break
        
    print("No Path Found!")

def print_path(end,pos,st):
    print(end, ",", pos,",",end='')
    while not st.is_empty():
        pos , nxt = st.pop()
        print(pos,",",end='')
    
    print(" ")