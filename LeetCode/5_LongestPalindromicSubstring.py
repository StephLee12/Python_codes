# palindromic 回文


# 思想：遍历每一个元素和两个元素之间的空位 并以该元素或该空位向两边扩散
# 自己写的
def v_1(s):
    substring = s[0]

    length = len(s)
    
    tmp_substring = None
    counter = 1
    for i in range(1,2*(length-1)):        
        
        if i % 2 == 1:
            if s[int(i/2)] != s[int(i/2)+1]:
                continue
            tmp_substring = s[int(i/2)]+s[int(i/2)+1]
            while (int(i/2)-counter) >= 0 and (int(i/2)+1+counter) <= length-1 and s[int(i/2)-counter] == s[int(i/2)+1+counter]:
                tmp_substring = s[int(i/2)-counter] + tmp_substring + s[int(i/2)+1+counter]
                counter += 1
            counter = 1
        elif i % 2 == 0:
            
            tmp_substring = s[int(i/2)]
            while (int(i/2)-counter >= 0) and ((int(i/2)+counter) <= (length-1)) and s[int(i/2)-counter] == s[int(i/2)+counter]:
                tmp_substring = s[int(i/2)-counter] + tmp_substring + s[int(i/2)+counter]
                counter += 1
            counter = 1
            
        if len(tmp_substring) > len(substring):
            substring = tmp_substring

    
    return substring

# 题解中动态规划 记录之前的信息 空间换时间
def v_2(s):
    length = len(s)

    if length < 2:
        return s
    
    dp = [[False for i in range(length)] for j in range(length)]

    for i in range(length):
        dp[i][i] = True
    
    start_pos = 0
    max_sublength = 1
    for j in range(1,length):
        for i in range(0,j):
            if s[i] == s[j]:
                if j-i < 3:
                    dp[i][j] = True
                else:
                    dp[i][j] = dp[i+1][j-1]
            
            if dp[i][j] == True:
                cur_sublength = j-i+1
                if cur_sublength > max_sublength:
                    max_sublength = cur_sublength
                    start_pos = i
    
    return s[start_pos:start_pos+max_sublength]



if __name__ == "__main__":
    v_1('ccc')