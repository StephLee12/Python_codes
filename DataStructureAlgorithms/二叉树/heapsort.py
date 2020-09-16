def heap_sort(elems):
    def shift_down(elems, elem , begin , end):
        i , j = begin , begin * 2 + 1

        while j < end:
            if (j + 1 < end ) and (elems[j+1] < elem[j]): #比较两个子结点的大小
                j =  j + 1
            
            if elem < elems[j]: #将两个子结点中较小的值 与elem比较
                break #elem直接就在堆顶 否则向下筛选
                
            elems[i] = elems[j]
            i , j = j , 2 * j + 1
        
        elems[i] = elem
    
    #heap_build
    end = len(elems)
    for i in range(end // 2, -1 ,-1):
        shift_down(elems,elems[i],i,end)
    
    for i in range((end-1),0,-1):
        e = elems[i]
        elems[i] = elems[0]
        shift_down(elems,e,0,i)