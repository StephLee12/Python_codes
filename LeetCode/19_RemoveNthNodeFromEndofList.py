# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


# 链表题常设置一个dummy节点 next指向head
# 自己写 AC
def v_1(head,n):
    p = head

    stack = []
    length = 0
    while p != None:
        stack.append(p)
        p = p.next
        length += 1
    
    if length-n == 0:
        return head.next
    else:
        nnode_prev = stack[length-n-1]
        nnode_prev.next = nnode_prev.next.next

    return head

# 快慢指针 注意dummy
def v_2(head,n):
    dummy = ListNode(0,head)
    fast_p = head
    slow_p= dummy
    for i in range(n):
        fast_p = fast_p.next 
    
    while fast_p != None:
        fast_p = fast_p.next
        slow_p = slow_p.next

    slow_p.next = slow_p.next.next 
    return dummy.next

    
