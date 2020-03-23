#插入排序
def insert_sort(lst):
    for i in range(1, len(lst)) : #开始时表的第一个元素不存在排序问题
        x = lst[i]
        j = i
        #将x前面比x大的元素 向后移动
        while j > 0 and lst[j-1].key > x.key: 
            lst[j] = lst[j-1]
            j = j-1
        #找到x的插入位置
        lst[j] = x 
#希尔排序
def shell_sort(lst):
    gap = len(lst) // 2 #//是取整
    while gap >= 1:

        for i in range(gap,len(lst)):
            tmp = lst[i]
            j = i 

            #进行插入排序
            while j >= gap and lst[j-gap] > tmp:
                lst[j] = lst[j-gap]
                j = j - gap
            
            lst[j] = tmp
        
        gap = gap // 2 #更新gap      
#选择排序
def select_sort(lst):
    #只需要循环len(lst-1)次，因为最后一个未排的元素不用动
    for i in range(len(lst) -1): 
        k = i #k是lst[i]后面的序列中已知的最小元素的位置
        for j in range(i,len(lst)):
            if lst[j].key < lst[k].key:
                k =  j
        #如果i!=k 即lst[i]后面的元素中存在比lst[i]更小的元素，进行交换。否则没必要交换
        if i != k: 
            lst[i] , lst[k] = lst[k], lst[i] 
#堆排序
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
        #交换堆顶元素和表尾端元素
        e = elems[i]  
        elems[i] = elems[0]
        #将表尾端元素放在堆顶 进行向下筛选
        #所以若是小顶堆 得到的排序序列是递减序列；若是大顶堆，得到递增序列
        shift_down(elems,e,0,i)
#冒泡排序
def bubble_sort(lst):
    for i in range(len(lst)):
        for j in range(1,len(lst) - i):
            if lst[j-1].key > lst[j].key:
                lst[j-1] , lst[j] = lst[j], lst[j-1]
#冒泡排序———改进
def bubble_sort_v2(lst):
    for i in range(len(lst)):
        flag = 0 #没有发现逆序 flag == 0
        for j in range(1, len(lst) -i):
            if lst[j-1].key > lst[j].key:
                lst[j-1] , lst[j] = lst[j], lst[j-1]
            flag = 1
        
        if flag == 0: #没有发现逆序
            break
#快速排序
def quick_sort_main(lst):
    qsort_rec(lst,0,len(lst)-1)

def qsort_rec(lst,l,r):
    if l >= r:
        return  #分段无数据或只有一个数据
    
    i ,j = l, r

    start = lst[i] #R
    while i < j:
        while i < j and lst[j].key >= start.key: #先从右边开始搜索
            j = j -1
        if i < j: #找到第一个 交换 i自增
            lst[i] = lst[j]
            i = i + 1
        while i < j and lst[i].key <= start.key: #再从左边搜索
            i = i  + 1
        if i < j: #找到第一个 交换 j自减
            lst[j] = lst[i]
            j = j -1 
    
    lst[i] = start # i==j时 存入R
    #进行递归
    qsort_rec(lst,l,i-1) #处理左部分
    qsort_rec(lst,i+1,r) #处理右部分
#归并排序
def merge_sort(lst):
    slen , llen = 1, len(lst)
    tmp_lst = [None] * llen
    while slen < llen:
        merge_pass(lst, tmp_lst, llen, slen)
        slen = slen * 2
        merge_pass(tmp_lst, lst, llen, slen) #进行交换
        slen = slen * 2

def merge_pass(lfrom,lto,llen,slen): #slen是要归并的序列的长度
    i = 0
    while i + 2 *slen <= llen : 
        merge(lfrom, lto ,i ,i+slen ,i + 2*slen)
        i = i + 2 * slen
    
    if i + slen < llen: #还剩两段，最后一段的长度小于slen
        merge(lfrom, lto, i , i + slen, llen)
    else: #只剩一段
         for j in range(i,llen):
             lto[j] = lfrom[j]

def merge(lfrom,lto,low,mid,high):
    i, j , k = low, mid , low
    while i < mid and j < high:
        if lfrom[i].key <= lfrom[j].key:
            lto[k] = lfrom[i]
            i = i + 1
        else: #lfrom[i].key >= lfrom[j].key
            lto[k] = lfrom[j]
            j = j + 1
        
        k = k + 1
    # 下面解决 len([low:mid]) != len([mid:high])的情况
    while i < mid: # len([low:mid])长度更长
        lto[k] = lfrom[i]
        i = i + 1
        k = k + 1
    while j < high: # len([mid:high])长度更长
        lto[k] = lfrom[j]
        j = j + 1
        k = k + 1
#基数排序/桶排序
def radix_sort(lst,d): #d是元组的长度
    rlists = [[] for i in range(10)] #创建桶 这里是十进制
    llen = len(lst)

    for m in range(-1 , -1 ,-d-1): #list中最后一个元素lst[-1] 倒数第二个lst[-2],即用LSD
        #将最低位存入桶中
        for j in range(llen):
            rlists[lst[j].key[m]].append(lst[j]) #lst[j].key[m]是lst[j]这个数据的key的最低位
        j = 0
        #顺序收集
        for i in range(10):
            tmp = rlists[i] #每个桶顺序收集
            for k in range(len(tmp)):
                lst[j] = tmp[k]
                j = j + 1
            rlists[i].clear() #清除桶 再次入桶