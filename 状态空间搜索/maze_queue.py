class QueueUnderflow(ValueError):
    pass

class SQueue():
    def __init__(self,init_len=8):
        self._len = init_len #存储区长度
        self._elems = [0] * init_len #元素存储 list对象
        self._head = 0     #队头
        self._num = 0  #队中元素个数
    
    def is_empty(self):
        return self._num == 0
    
    def peek(self):
        if self._num == 0:
            raise QueueUnderflow("in peek")

        return self._elems[self._head]
    
    def dequeue(self):
        if self._num == 0:
            raise QueueUnderflow("in dequeue") 

        e =  self._elems[self._head]
        self._head = (self._head + 1) % self._len
        self._num = self._num -1 
        return e
    
    def enqueue(self,elem):
        if self._num == self._len:
            self.__extend()
        
        self._elems[(self._head + self._num) % self._len ] = elem
        self._num  = self._num  + 1
    
    def __extend(self):
        old_len = self._len
        self._len = self._len * 2
        new_elems = [0] * self._len

        for i in range(old_len):
            new_elems[i] = self._elems[(self._head + i) % old_len]
        
        self._elems, self._head = new_elems, 0

dirs = [(0,1),(1,0),(0,-1),(-1,0)]

def mark(maze,pos): #标记搜索过的位置为2
    maze[pos[0]][pos[1]] == 2

def passable(maze,pos): #判断位置pos是否可以通行
    return maze[pos[0]][pos[1]] == 0

def maze_que(maze,start,end):
    if start == end:
        print("Start Point is the end")
        return

    qu = SQueue()
    mark(maze,start)
    qu.enqueue(start)

    while not qu.is_empty():
        pos = qu.dequeue()

        for i in range(4):
            nextp = (pos[0] + dirs[i][0] ,
                        pos[1] + dirs[i][1])

            if passable(maze,nextp):
                if nextp == end:
                    print("Problem Solved")
                    return

                mark(maze, nextp)
                qu.enqueue(nextp)

    print("No Solution") 