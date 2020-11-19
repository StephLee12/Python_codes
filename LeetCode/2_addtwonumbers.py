# Definition for singly-linked list.
class ListNode:
    def __init__(self, x):
        self.val = x
        self.next = None

def addTwoNumber(l1,l2):
    l1_pointer,l2_pointer = l1,l2
    res_llist= ListNode(0)# 要返回的链表
    p = res_llist # 链表指针
    forward_signal = 0 # 进位信号

    while l1_pointer != None and l2_pointer != None:
        l1_l2_sum = l1_pointer.val + l2_pointer.val
        # 先考虑前一位计算是否进位
        if forward_signal == 1:
            # 若前一位进位后 该位需要进位
            if (l1_l2_sum + forward_signal) // 10 == 1:
                # 依然需要进位 则进位信号保持不变
                p.val = (l1_l2_sum + forward_signal) % 10
            else: # 前一位进位后改为不需要进位 进位后 forward_signal置0
                p.val = l1_l2_sum + forward_signal
                forward_signal = 0           
            
        # 若前一位不进位
        else: 
            # 考虑下一位需要进位的情况
            if l1_l2_sum // 10 == 1:
                forward_signal = 1 # 进位信号置1 
                p.val = l1_l2_sum % 10
            else:
                p.val = l1_l2_sum

        l1_pointer,l2_pointer = l1_pointer.next ,l2_pointer.next
        if l1_pointer == None or l2_pointer == None:
            continue
        else:
            p.next = ListNode(0)
            p = p.next # 指针向前
        
    # 若两个链表长度不等
    if l1_pointer == None and l2_pointer != None: # 若l1链表已遍历完
        p.next = ListNode(0)
        p = p.next
        while l2_pointer != None:
            # 考虑是否进位
            if forward_signal == 1:
                if (l2_pointer.val + forward_signal) // 10 == 1:
                    p.val = (l2_pointer.val + forward_signal) % 10
                else: #进位后 forward_signal置0
                    p.val = l2_pointer.val + forward_signal
                    forward_signal = 0
            else:
                p.val = l2_pointer.val
            
            l2_pointer = l2_pointer.next
            if l2_pointer == None: # 先判断循环是否结束
                continue
            else:
                # 如果循环不结束再创建新的节点
                p.next = ListNode(0)
                p = p.next
    elif l2_pointer == None and l1_pointer != None: # 若l2链表已遍历完
        p.next = ListNode(0)
        p = p.next
        while l1_pointer != None:
            # 考虑是否进位
            if forward_signal == 1:
                if (l1_pointer.val + forward_signal) // 10 == 1:
                    p.val = (l1_pointer.val + forward_signal) % 10
                else: #进位后 forward_signal置0
                    p.val = l1_pointer.val + forward_signal
                    forward_signal = 0

            else:
                p.val = l1_pointer.val

            l1_pointer = l1_pointer.next
            if l1_pointer == None:
                continue
            else:
                p.next = ListNode(0)
                p = p.next
    
    # 若最后还有一个进位信号
    if forward_signal == 1:
        p.next = ListNode(1)

    return res_llist


if __name__ == "__main__":
    l1  = ListNode(2)
    l1.next = ListNode(4)
    l1.next.next = ListNode(3)
    l2 = ListNode(5)
    l2.next = ListNode(6)
    l2.next.next = ListNode(4)
    res = addTwoNumber(l1,l2)
    while res != None:
        print(res.val)
        res = res.next

        