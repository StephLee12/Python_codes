class PriQueueError(ValueError):
    pass

class PriQueue:
    def __init__(self, elist=[]):
        self._elems = list(elist)
        if elist:
            self.buildheap()
    
    def is_empty(self):
        return not self._elems
    
    def peak(self): #获取堆顶元素
        if self.is_empty():
            raise PriQueueError("in peak")
        return self._elems[0]

    def enqueue(self,elem):
        self._elems.append(None) #插入一个空的元素

        self.shift_up(elem, len(self._elems)-1 )  #序号是从0开始编号 因此要长度减去1
    
    def shift_up(self,elem,last):
        elems, i, j = self._elems , last , (last -1 )/2  #完全二叉树的性质 父结点的编号为 (last-1)/2

        while (i > 0) and (elem < elems[j]): #不断向上比较
            elems[i] = elems[j] #将父结点的值赋给子结点 不断迭代
            i, j = j, (j - 1) /2  #更新子结点i的编号 和父结点j的编号
        
        elems[i] = elem

    def dequeue(self):
        if self.is_empty():
            raise PriQueueError("in dequeue")
        
        elems = self._elems
        heap_top = elems[0]
        heap_last = elems[-1]
        if len(elems) > 0:
            self.shift_down(heap_last, 0 , len(elems))

        return heap_top

    def shift_down(self, elem , begin , end):
        elems , i , j = self._elems, begin , begin * 2 + 1

        while j < end:
            if (j + 1 < end ) and (elems[j+1] < elem[j]): #比较两个子结点的大小
                j =  j + 1
            
            if elem < elems[j]: #将两个子结点中较小的值 与elem比较
                break #elem直接就在堆顶 否则向下筛选
                
            elems[i] = elems[j]
            i , j = j , 2 * j + 1
        
        elems[i] = elem
        
    def buildheap(self): #构建初始的堆
        end =  len(self._elems)
        for i in range(end // 2 , -1 , -1 ): #从下标 end/2取整开始都是叶结点
            self.shift_down(self._elems[i], i , end)
