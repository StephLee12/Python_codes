class LinkedListUnderflow(ValueError): #表是否溢出
    pass

class DLNode:
          def __init__(self,elem,prev_=None,next_=None): #创建一个新结点
            self.elem = elem
            self.prev = prev_
            self.next = next_

class DLList:

    def __init__(self):
        self._head = None
    
    def is_empty(self):
        return self._head is None
    
    def prepend(self,elem):
        if self.is_empty():
            self._head = DLNode(elem,None,self._head)
        
        else:
            self._head = DLNode(elem,None,self._head)
            self._head.next.prev = self._head
    
    def append(self,elem):
        if self.is_empty():
            self._head = DLNode(elem,None,self._head)
        
        else:
            p = self._head

            while p.next is not None:
                p = p.next
            
            e = DLNode(elem,p,None)
            p.next = e
    
    def length(self):
        if self.is_empty():
            return 0
        
        else:
            p = self._head
            counter = 0

            while p is not None:
                p = p.next
                counter = counter + 1 
            
            return counter
    
    def pop(self):
        if self.is_empty():
            raise LinkedListUnderflow("in pop")

        elif self.length() == 1:
            e = self._head.elem
            self._head = None
            return e
        
        else:
            e = self._head.elem
            self._head = self._head.next
            self._head.prev = None
            return e
    
    def pop_last(self):
        if self.is_empty():
            raise LinkedListUnderflow("in pop_last")

        elif self.length() == 1:
            e = self._head.elem
            self._head = None
            return e
        
        else:
            p = self._head

            while p.next.next is not None:
                p = p.next
            
            e = p.next.elem
            p.next.prev = None
            p.next = None

            return e
    
    def print_list(self):
        if self.is_empty():
            raise LinkedListUnderflow("in print")

        else:
            p = self._head

            while p is not None:
                print(p.elem,end='')

                if p.next is not None:
                    print(', ',end='')
                
                p = p.next
            print('')
        
    def print_reverse(self): #用来检查双链表的正确性
        if self.is_empty():
            raise LinkedListUnderflow("in print_reverse")

        else:
            p = self._head

            while p.next is not None:
                p = p.next
            
            while p is not None:
                print(p.elem,end='')
                if p.prev is not None:
                    print(', ',end='')
                p = p.prev
            print('')

            

    
    def insert(self,pos,elem):
        if pos == 0:
            self.prepend(elem)
        elif pos == self.length():
            self.append(elem)
        else:
            p = self._head
            counter = 0

            while counter < pos - 1 :
                counter = counter + 1 
                p = p.next

            e = DLNode(elem,p,p.next)
            p.next.prev = e
            p.next = e

    def delete(self, pos):
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
            p.next = p.next.next
            p.next.prev = p

            return e
    
    
        
if __name__ == "__main__":
    
    dlist1 = DLList()

    dlist1.prepend(1)

   
    #dlist1.print()

    for i in range (2,10):
        dlist1.prepend(i)
    
    #dlist1.print()
    #print(dlist1.length())

    #print(dlist1.pop_last())
    #dlist1.insert(1,89)
    dlist1.delete(2)
    dlist1.print_list()
    dlist1.print_reverse()
   
            
                



        
