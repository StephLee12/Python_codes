# 思路太乱了
def v_1(s,p):
    def judge_none(s):
        head = s[0]

        if len(s) == 1:
            if head == '*':
                return True
            else:
                return False
        
        if head == '*':
            s = s[1:len(s)]
            if len(s) % 2 == 1:
                return False
            else:
                for i in range(0,len(s)-1 ,2):
                    if s[i+1] != '*':
                        return False
            
        else:
            if len(s) % 2 == 1:
                return False
            for i in range(0,len(s)-1,2):
                if s[i+1] != '*':
                    return False
        return True

    
    s_len = len(s)
    p_len = len(p)

    s_pointer = 0
    p_pointer = 0

    dp = [[True for i in range(s_len)] for j in range(p_len)]

    while s_pointer < s_len:
           
        if p[p_pointer] == '*':
            if dp[p_pointer-1][s_pointer] == False:
                p_pointer += 1
            elif dp[p_pointer-1][s_pointer-1] == True:
                p_elem = p[p_pointer-1]
                s_elem = s[s_pointer-1]
                if p_pointer +1 == p_len and s_pointer + 1 == s_len: # 边界情况
                    return True
                if p_pointer + 1 == p_len: # 边界情况
                    if p_elem == '.':
                        return True
                    else:                       
                        if judge_none(s[s_pointer+1:s_len]) == True:
                            return True
                        counter = 0
                        for i in range(s_pointer+1,s_len):
                            if s[i] == p_elem:
                                counter += 1
                        if counter == s_len - s_pointer -1:
                            return True
                        else:
                            return False
                
                p_rest = p[p_pointer+1:p_len]
                is_non_p_rest = judge_none(p_rest)
                if s_pointer + 1 == s_len: # 边界情况
                    s_last = s[s_pointer]
                    if v_1(s_last,p_rest) == True or is_non_p_rest == True:
                        return True
                    else:
                        return False
                s_rest = s[s_pointer+1:s_len]
                s_rest_len = s_len-s_pointer-1
                
                # 普遍情况
                if p_elem != '.':
                    if s[s_pointer] != s_elem:
                        p_pointer += 1
                        continue   

                    if v_1(s_rest,p_rest) == True:
                        return True
                    else:    
                        s_pointer += 1        
                        while s_pointer < s_len and s[s_pointer] == p_elem:
                            s_pointer += 1
                        p_pointer += 1
                        dp[p_pointer-1][s_pointer-1] = True

                else:
                    counter = 0
                    tmp_rest = s_rest[s_rest_len-1]
                    while len(tmp_rest) == s_rest_len:
                        if v_1(tmp_rest,p_rest) == True:
                            return True
                        else:
                            counter += 1
                            tmp_rest += s_rest[s_rest_len-1-counter]
                    return False
           
            if p_pointer == p_len and s_pointer < s_len:
                return False
            continue

        if s[s_pointer] != p[p_pointer] and p[p_pointer] != '.':
            dp[p_pointer][s_pointer] = False
            p_pointer += 1
            if p_pointer == p_len and s_pointer < s_len:
                return False
            continue          

        if p[p_pointer] == '.' or s[s_pointer] == p[p_pointer]:
            dp[p_pointer][s_pointer] == True
            p_pointer += 1
            s_pointer += 1
            if p_pointer == p_len and s_pointer < s_len:
                return False
            continue

        if dp[p_pointer-1][s_pointer-1] == False:
            return False
    
    if p_pointer != p_len:
        p_rest = p[p_pointer:p_len]
        if judge_none(p_rest) == True:
            return True
        else:
            return False
    
    return True

# 看题解
def v_2(s,p):

    if not p : return not s 
    if not s and len(p) == 1: return False
    
    row_shape = len(s) + 1
    col_shape = len(p) + 1

    # 初始化 ！！ 很精髓
    dp = [[False for c in range(col_shape)] for r in range(row_shape)]
    
    dp[0][0] = True
    dp[0][1] = False
    for c in range(2,col_shape):
        j = c-1
        if p[j] == '*': dp[0][c] = dp[0][c-2]
    
    for r in range(1,row_shape):
        i = r-1
        for c in range(1,col_shape):
            j = c-1
            if s[i] == p[j] or p[j] == '.': 
                dp[r][c] = dp[r-1][c-1]
            elif p[j] == '*':
                if p[j-1] == s[i] or p[j-1] == '.':
                    dp[r][c] = dp[r-1][c] or dp[r][c-2]
                else:
                    dp[r][c] = dp[r][c-2]
            else:
                dp[r][c] = False
    
    return dp[row_shape-1][col_shape-1]



if __name__ == "__main__":
    v_1('bbbba','.*a*a')