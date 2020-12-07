# 自己写的
def v_1(s):
    sub_length = 0
    str_length = len(s)

    stack = []
    counter = 0
    while counter < str_length:

        next_elem = s[counter]

        equal_flag = 0
        last_equal_idx = 0
        for i,elem in enumerate(stack):
            if elem == next_elem:
                equal_flag = 1
                last_equal_idx = i

        if equal_flag == 1:
            stack = stack[last_equal_idx + 1:]
        stack.append(next_elem)
        stack_size = len(stack)
        if stack_size > sub_length:
            sub_length = stack_size
                
        counter += 1

    # sub_length = 0
    # str_length = len(s)

    # stack = []
    # counter = 0
    # pointer = 0
    # while counter < str_length:

    #     while pointer < str_length and s[pointer] not in stack:
    #         stack.append(s[pointer])
    #         pointer += 1
    #     sub_length = max(sub_length,pointer-1-counter+1)
    #     stack.pop(0)
    #     counter += 1

    return sub_length


def v_2(s):

    sub_length = 0
    str_length = len(s)

    stack = []
    counter = 0
    pointer = 0
    while counter < str_length:

        while pointer < str_length and s[pointer] not in stack:
            stack.append(s[pointer])
            pointer += 1
        sub_length = max(sub_length,pointer-1-counter+1)
        stack.pop(0)
        counter += 1
        
# 用set 滑动窗口
def v_3(s):
    max_sub_length = 0
    str_length = len(s)

    max_substring = set()
    pointer = -1

    for i in range(str_length):
        if i != 0:
            max_substring.remove(s[i-1])
        while (pointer + 1 < str_length) and (s[pointer+1] not in max_substring):
            max_substring.add(s[pointer+1])
            pointer += 1
        max_sub_length = max(max_sub_length,pointer-i+1)
    
    return max_sub_length
