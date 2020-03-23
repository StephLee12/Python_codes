class LinkedListUnderflow(ValueError): #表是否溢出
    pass

class LNode:
          def __init__(self,elem,next_=None): #创建一个新结点
            self.elem = elem
            self.next = next_

class LCList:
    
    def __init__(self): #创建空表
        self._head = None

    def is_empty(self): #判断空表
        return self._head is None
    
    def prepend(self,elem): #表首添加元素
        if self.is_empty():
            self._head = LNode(elem,self._head)
            self._head.next = self._head
        
        else:
            
            p = self._head

            while p.next != self._head:
                p = p.next
            
            e = LNode(elem,self._head)
            p.next = e
            self._head = e
    
    def length(self):  #获取表的长度
        
        if self.is_empty():
            return 0

        else:
            p = self._head
            counter = 1

            while p.next != self._head:
                counter = counter + 1 
                p = p.next
            return counter

    def append(self,elem): #表尾添加元素
        if self.is_empty():
            self._head = LNode(elem,self._head)
            self._head.next = self._head
        
        else:
            p = self._head

            while p.next != self._head:
                p = p.next

            e = LNode(elem,self._head)
            p.next = e

    def pop(self): #删除表首元素
        if self.is_empty():
            raise LinkedListUnderflow("in pop")

        elif self._head.next == None:
            
            e = self._head.elem
            self._head = None

            return e
        
        else:

            p = self._head

            while p.next != self._head:
                p = p.next
            
            e = self._head.elem
            p.next = self._head.next
            self._head = self._head.next
            
            return e
    
    def pop_last(self):
        if self.is_empty():
            raise LinkedListUnderflow("in pop last")

        elif self._head.next == None:
            e = self._head.elem
            self._head = None

            return e
        
        else:
            p = self._head

            while p.next.next != self._head:
                p = p.next
            
            e = p.next.elem
            p.next = self._head
            
            return e
    
    def print(self):
        if self.length() < 1:
            raise LinkedListUnderflow("in print")

        else:
            p = self._head

            while p.next != self._head:
                print(p.elem,end='')
                print(', ',end='')
                p = p.next
            
            print(p.elem)
            print('')
    
    def insert(self,pos,elem):
        if pos == 0:
            self.prepend(elem)
        elif pos == self.length():
            self.append(elem)
        elif pos < 0 | pos > self.length():
            raise LinkedListUnderflow("in insert")
        else:

            p = self._head
            counter = 0

            while counter < pos -1:
                counter = counter + 1 
                p = p.next

            e = LNode(elem,p.next)
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
            p.next = p.next.next

            return e

if __name__ == "__main__":
    
    lclist1 = LCList()

    #lclist1.prepend(1)

    #print(lclist1._head)

    #lclist1.prepend(2)

    #print(lclist1.length())

    #lclist1.print()

    #lclist1.append(3)

    #lclist1.append(4)

    #lclist1.print()

    #print(lclist1.pop_last())

    #lclist1.print()

    #lclist1.insert(1,6)

    #lclist1.print()

    for i in range(10):
        lclist1.append(i + 1)

    lclist1.print()

    print(lclist1.delete(8))
    print(lclist1.length())
    lclist1.print()
    print(lclist1.delete(9))



    




    


            



        
            
