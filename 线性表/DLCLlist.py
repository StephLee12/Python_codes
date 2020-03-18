class LinkedListUnderflow(ValueError): #表是否溢出
    pass

class DLNode:
          def __init__(self,elem,prev_=None,next_=None): #创建一个新结点
            self.elem = elem
            self.prev = prev_
            self.next = next_

class DCLList:

    def __init__(self):
        self._head = None
    
    def is_empty(self):
        return self._head is None

    def prepend(self,elem):
        if self.is_empty():
            self._head = DLNode(elem,None,None)
            self._head.next = self._head
            self._head.prev = self._head
        else:
            last = self._head.prev
            e = DLNode(elem,last,self._head)
            last.next = e
            self._head.prev = e
            self._head = e
    
    def append(self,elem):
        if self.is_empty():
            self._head = DLNode(elem,None,None)
            self._head.next = self._head
            self._head.prev = self._head
        
        else:
            last = self._head.prev
            e = DLNode(elem,last,self._head)
            last.next = e
            self._head.prev = e
    
    def length(self):
        if self.is_empty():
            return 0
        
        else:
            p = self._head
            counter = 0
            while p.next is not self._head:
                counter = counter + 1 
                p = p.next
            counter = counter + 1
            return counter 
    
    def print_list(self):
        if self.is_empty():
            raise LinkedListUnderflow("in print_list")

        else:
            p = self._head
            while p.next is not self._head:
                print(p.elem,end='')
                print(', ',end='')
                p = p.next
            print(p.elem)
            print('')
    
    def pop(self):
        if self.is_empty():
            raise LinkedListUnderflow("in pop")
        
        else:
            e = self._head.elem
            self._head.prev.next = self._head.next
            self._head.next.prev = self._head.prev
            self._head = self._head.next
            return e
    
    def pop_last(self):
        if self.is_empty():
            raise LinkedListUnderflow("in pop_last")
        
        else:
            last_forward = self._head.prev.prev
            e = self._head.prev.elem
            last_forward.next = self._head
            self._head.prev = last_forward
            return e
    
    def insert(self,pos,elem):
        if pos == 0 :
            self.prepend(elem)
        elif pos == self.length():
            self.append(elem)
        else:
            p = self._head
            counter = 0
            while counter < pos - 1:
                counter = counter + 1
                p = p.next
            
            e = DLNode(elem,p,p.next)
            p.next.prev = e
            p.next = e
    
    def delete(self,pos):
        if pos < 1 | (pos > self.length()):
            raise LinkedListUnderflow("in delete")
        elif pos == 1:
            return self.pop()
        elif pos == self.length():
            return self.pop_last()
        else:
            p = self._head
            counter = 1 
            while counter < pos -1 :
                counter = counter + 1
                p = p.next
            
            e = p.next.elem
            p.next.next.prev = p
            p.next = p.next.next
            
            return e
if __name__ == "__main__":
    
    dcllist1 = DCLList()

    for i in range(1,10):
        dcllist1.prepend(i)
        
    #dcllist1.print_list()
    #print(dcllist1.length())

    for i in range(11,20):
        dcllist1.append(i)
    #dcllist1.print_list()

    print(dcllist1.pop())
    print(dcllist1.pop_last())
    #dcllist1.print_list()
    dcllist1.insert(3,100)
    dcllist1.print_list()
    dcllist1.delete(4)
    dcllist1.print_list()


        
            