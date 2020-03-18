class LinkedListUnderflow(ValueError): #表是否溢出
    pass

class LNode:
          def __init__(self,elem,next_=None): #创建一个新结点
            self.elem = elem
            self.next = next_

def list_sort(lst):
    for i in range(1,len(lst)):
        x = lst[i]
        j = i 
        while j > 0 & lst[j-1] > x:
            lst[j] = lst[j-1]
            j = j -1 
        
        lst[j] = x

class LList:
    
    def __init__(self): #创建空表
        
        self._head = None
    
    def is_empty(self): #判断表是否为空
        
        return self._head is None
    
    def length(self):
        
        p = self._head
        counter = 0

        while p is not None:
            
            counter = counter + 1 
            p = p.next
        
        return counter
        
    def prepend(self,elem): #在表首加入元素
        
        self._head = LNode(elem,self._head)

    def append(self, elem): #在表尾加入元素
            
            if self.is_empty():
                self._head = LNode(elem)

            p = self._head

            while p.next is not None:
                p = p.next
            
            p.next = LNode(elem)
            
    def pop(self): #删除表首元素
        
        if self.is_empty():
            raise LinkedListUnderflow("in pop")
            
        
        e = self._head.elem
        self._head =  self._head.next

        return e

    def pop_last(self): #删除表尾元素
        
        if self.is_empty(): #空表
            raise LinkedListUnderflow("in pop_last")
            
        
        p = self._head

        if p.next is None: #表只有一个结点
            e = p.elem
            self._head = None
            return e
        
        while p.next.next is not None: #找到倒数第二个结点 即p.next是表尾结点
            p = p.next
        
        e = p.next.elem #弹出最后一个结点的元素内容
        p.next = None
        
        return e
    
    def find(self,elem): #查找链表中的第一个elem元素
        
        p = self._head
        
        pos = 0

        while p is not None:
            
            pos = pos + 1
            if p.elem == elem :
                return pos #如果找到第一个元素 返回位置
            else:
                p = p.next
        
        return -1 #没有找到 返回-1
    
    def printall(self): #打印所有结点
        
        p = self._head
        
        while p is not None:
            print(p.elem, end='')
            if p.next is not None:
                print(', ',end='')
            p = p.next
        
        print('')

    def for_each(self,proc): #表的遍历
        
        p = self._head
        
        while p is not None:
            proc(p.elem)
            p = p.next
    
    def elements(self): #生成器函数
        
        p = self._head
        
        while p is not None:
            yield p.elem
            p = p.next
    
    def filter(self,pred): #筛选生成器函数
        
        p = self._head
        
        while p is not None:
            if pred(p.elem):
                yield p.elem
            p = p.next
    
    def insert_general(self,pos,elem):
        
        if pos < 1:
            self.prepend(elem)
        
        elif pos > self.length():
            self.append(elem)
        
        else:
            p = self._head

            counter = 0

            while counter < pos - 1 :
                counter = counter + 1 
                p = p.next
            
            new_node = LNode(elem,p.next)
            p.next = new_node
    
    def pop_general(self,pos):

        if (pos < 1) | (pos > self.length()) :
            raise LinkedListUnderflow("in pop_general")
            
        
        elif pos == 1:
            return self.pop()
        
        elif pos == self.length():
            return self.pop_last()
        
        else : 

            p = self._head

            counter = 1

            while counter < pos -1 :

                counter = counter + 1

                p = p.next
            
            e = p.next.elem

            p.next = p.next.next

            return e
    
    def list_reverse(self):
        if self.is_empty():
            raise LinkedListUnderflow("in list_reverse")
        else:
            p = None

            while self._head is not None:
                tmp = self._head
                self._head = tmp.next
                tmp.next = p
                p = tmp
            
            self._head = p
    
    def sort_one(self): #相当于是将一个元素插入 其他的依次向后移动
        
        if self.is_empty():
            raise LinkedListUnderflow("in sort")
        elif self.length() == 1:
            return 
        
        else:
            scanPointer = self._head.next

            while scanPointer is not None:
                
                p = self._head
                scanElem = scanPointer.elem

                while p is not scanPointer and p.elem <= scanElem:
                    p =  p.next
                
                while p is not scanPointer:
                    
                    tmp = p.elem
                    p.elem = scanElem
                    scanElem = tmp
                    p = p.next
                
                scanPointer.elem = scanElem
                scanPointer = scanPointer.next

    def sort_two(self): #一个元素插入 只需要改变链接 #把p scanPointer理解为指向结点的指针
        if self.is_empty():
            raise LinkedListUnderflow("in sort_two")
        elif self.length() == 1:
            return 
        else:
            p = self._head
            scanPointer = p.next
            scanElem = scanPointer.elem
            
            p.next = None 
            q = None

            while scanPointer is not None:
                
                p = self._head

                while p is not None and p.elem <= scanElem:
                    q = p
                    p = p.next
                
                if q is None:
                    self._head = scanPointer
                
                else :
                    q.next = scanPointer
                
                q = scanPointer
                scanPointer  = scanPointer.next
                q.next = p

        
if __name__ == "__main__":
    
    mlist1 = LList()

    for i in range(10):
        mlist1.prepend(i)
    
    for i in range(11,20):
        mlist1.append(i)

#    mlist1.printall()

#    print(mlist1.length())

    mlist1.insert_general(100,3)

#    mlist1.printall()

#    print(mlist1.length())

#    print(mlist1.find(8))

    mlist1.pop()

    mlist1.pop_last()
    
    mlist1.printall()

    print(mlist1.pop_general(4))

    mlist1.printall()

    mlist1.insert_general(3,6)

    mlist1.printall()

    mlist1.sort_two()

    mlist1.printall()

