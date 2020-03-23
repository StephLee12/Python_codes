class LNode:
          def __init__(self,elem,next_=None):
            self.elem = elem
            self.next = next_

if __name__ == "__main__":
    llist1 = LNode(1) #先创建一个LNode的对象 了llist1

    p = llist1

    for i in range(2,11):
        p.next = LNode(i) 
        p = p.next
    
    p = llist1
    while p is not None:
        print(p.elem)
        p = p.next