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

class LNode:
          def __init__(self,elem,next_=None):
            self.elem = elem
            self.next = next_

class LStack(): #基于链接表技术实现的栈类，基于单链表
    
    def __init__(self):
        self._top = None
    
    def is_empty(self):
        return self._top is None
    
    def top(self):
        if self._top is None:
            raise StackUnderflow("in LStack.top()")
        return self._top.elem
    
    def push(self,elem):
        self._top = LNode(elem,self._top)
    
    def pop(self):
        if self._top is None:
            raise StackUnderflow("in LStack.pop()")

        e = self._top.elem
        self._top = self._top.next

        return e