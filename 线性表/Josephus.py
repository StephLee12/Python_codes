# n总人数 k从第k个人开始报数 m报到第m个 出列 从m的下一个开始
def josephus_A_mine(n, m, k):

    list1 = list(range(1, n + 1))

    start = k - 1

    flag = int(0)

    for i in range(n):

        counter = 0
        tmp = m
        flag = 0

        while counter < tmp:

            if list1[start + counter] > 0:
                counter = counter + 1

            elif list1[start + counter] == 0:
                counter = counter + 1
                tmp = tmp + 1

            #到了list的尾部进行讨论
            if start + counter > n - 1 and counter != tmp:
                start = 0
                tmp = tmp - counter
                counter = 0
                flag = 0
            elif start + counter > n - 1 and counter == tmp:
                flag = 1

        print(list1[start + counter - 1], end='')

        if i < n - 1:
            print(', ', end='')
        else:
            print('')
        list1[start + counter - 1] = 0
        if flag == 0:
            start = start + counter
        if flag == 1:
            start = 0

def josephus_A_book(n, m, k):
    people = list(range(1, n + 1))

    i = k - 1
    for num in range(n):
        count = 0
        while count < m:
            if people[i] > 0:
                count = count + 1
            if count == m:
                print(people[i], end='')
                people[i] = 0
            i = (i + 1) % n  #这个也太猛了
        if num < n - 1:
            print(', ', end='')
        else:
            print('')

def josephus_B_book(n, m, k):
    people = list(range(1, n + 1))

    num, i = n, k - 1

    #num这里是表的长度 每执行一次循环 表的长度减一
    for num in range(n, 0, -1):
        i = (i + m - 1) % num
        print(people.pop(i), end=(', ' if num > 1 else '\n'))

class LinkedListUnderflow(ValueError):  #表是否溢出
    pass

class LNode:
    def __init__(self, elem, next_=None):  #创建一个新结点
        self.elem = elem
        self.next = next_

class LCList:
    def __init__(self):  #创建空表
        self._head = None

    def is_empty(self):  #判断空表
        return self._head is None

    def prepend(self, elem):  #表首添加元素
        if self.is_empty():
            self._head = LNode(elem, self._head)
            self._head.next = self._head

        else:

            p = self._head

            while p.next != self._head:
                p = p.next

            e = LNode(elem, self._head)
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

    def append(self, elem):  #表尾添加元素
        if self.is_empty():
            self._head = LNode(elem, self._head)
            self._head.next = self._head

        else:
            p = self._head

            while p.next != self._head:
                p = p.next

            e = LNode(elem, self._head)
            p.next = e

    def pop(self):  #删除表首元素
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
                print(p.elem, end='')
                print(', ', end='')
                p = p.next

            print(p.elem)
            print('')

    def insert(self, pos, elem):
        if pos == 0:
            self.prepend(elem)
        elif pos == self.length():
            self.append(elem)
        elif pos < 0 | pos > self.length():
            raise LinkedListUnderflow("in insert")
        else:

            p = self._head
            counter = 0

            while counter < pos - 1:
                counter = counter + 1
                p = p.next

            e = LNode(elem, p.next)
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

            while counter < pos - 1:
                counter = counter + 1
                p = p.next

            e = p.next.elem
            p.next = p.next.next

            return e

class Josephus_C_mine(LCList):
    def __init__(self, n, m, k):
        LCList.__init__(self)
        for i in range(n):
            self.append(i + 1)

        p = self._head
        counter = 0
        loopCounter = 0

        while counter < k:
            counter = counter + 1
            if counter == k:
                break
            else:
                p = p.next

        for i in range(n):
            tmp = 1
            while tmp < m:
                tmp = tmp + 1
                p = p.next
                counter = counter + 1
                if tmp == m:
                    p = p.next
                    print(self.delete(counter), end='')
                    if counter == n - loopCounter :
                        counter = 1
                    
            loopCounter = loopCounter + 1


if __name__ == "__main__":
    josephus_A_book(10, 2, 7)
    josephus_A_mine(10, 2, 7)
    josephus_B_book(10, 2, 7)

    Josephus_C_mine(10, 2, 7)
