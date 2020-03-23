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