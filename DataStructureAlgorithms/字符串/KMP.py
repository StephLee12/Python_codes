def matching_KMP(t,p,pnext):
    i,j = 0,0
    m,n = len(p),len(t)

    while j < n and i < m :
        if i == -1 or p[i] == t[j]:
            i,j = i+1,j+1
        else:
            i = pnext[i]
    if i == m:
        return j - 1 #匹配成功 返回下标

    return -1 #没有匹配成功，返回特殊值
    
def gen_pnext(p):
    i ,k ,m = 0 , -1 , len(p)
    pnext = [-1] * m #
    
    while i < m -1:
        if k == -1 or p[i] == p[k]:
            i, k  = i+1, k+1
            pnext[i] = k
        else:
            k = pnext[k]

    return pnext