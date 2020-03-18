from assoc import Assoc
from bintreenode import BinTNode
from stack import SStack

class DictBinTree: 
    def __init__(self,root=None): #每个结点都是一个Assoc对象
        self._root = root
    
    def is_empty(self):
        return self._root is None
    
    def search(self,key):
        bt = self._root
        while bt is not None:
            entry = bt.data
            if entry.key < key:
                bt = bt.right
            elif entry.key > key:
                bt = bt.left
            else:
                return entry.value
        return None
    
    def insert(self,Assoc):
        key , value = Assoc.key, Assoc.value
        bt = self._root

        if bt is None:
            self._root = BinTNode(Assoc)
            return 

        while bt is not None:
            entry = bt.data
            
            if key < entry.key:
                if bt.left is None:
                    bt.left = BinTNode(Assoc)
                    return 
                
                bt = bt.left
            elif key > entry.key:
                if bt.right is None:
                    bt.right = BinTNode(Assoc)
                    return 
                bt = bt.right
            
            else:
                entry.value = value
                return 
    
    def entries(self): #中序遍历
        bt , st = self._root, SStack()

        while bt is not None or not st.is_empty():
            while bt is not None:
                st.push(bt)
                bt = bt.left
            
            bt = st.pop()
            yield bt.data.key, bt.data.value
            bt = bt.right

    def delete_v1(self,key):
        p, q = None, self._root

        while q is not None and q.data.key != key:
            p = q
            if key < q.data.key:
                q = q.left
            
            else:
                q = q.right
        
        if q is None:
            return False
        
        if q.left is None: #这个分支包含了q是叶结点和q只有右子树的情况
            if p is None : #删除的q是根结点
                self._root = q.right
            
            elif q == p.left:
                p.left = q.right
            else:
                p.right = q.right
            return
        
        #包含了q只有左子树和左右子树都有的情况
        r = q.left
        while r.right is not None:
            r = r.right
        
        r.right = q.right
        
        if p is None:
            self._root = q.right
        elif q == p.left:
            p.left = q.left 
        else:
            p.right = q.left 

    def delete_v2(self,key):
        p, q = None, self._root

        while q is not None and q.data.key != key: #找到要删除的结点
            p = q
            if key < q.data.key:
                q = q.left
            
            else:
                q = q.right
        
        if q is None: #没找到要删除的结点
            return False
        
        if q.left is None: #这个分支包含了q是叶结点和只有右子树的情况
            if p is None : #删除的q是根结点
                self._root = q.right
            
            elif q == p.left:
                p.left = q.right
            else:
                p.right = q.right
            return
        
        #v2只修改下面的部分
        if q.left is not None and q.right is None: #q只有左子树
            if p is None:
                self._root = q.left 
            elif q == p.left :
                p.left =  q.left
            else:
                p.right = q.left 

        ## q两棵子树都有
        prev_r, r = q, q.right

        if r.left is None: #q的右子树没有左子树
            if p is None:
                self._root = r
                self._root.left = prev_r.left
            elif q == p.left :
                p.left = r
                r.left = q.left 
            else:
                p.right = r
                r.left = q.left

        while r.left is not None: #找右子树中最小的
            prev_r = r
            r = r.left
        
        if r.right is not None: #最小元素有右子树
            if p is None: #删除结点是根结点
                prev_r.left = r.right
                self._root = r
                self._root.left = q.left
                self._root.right = q.right
            else:
                prev_r.left = r.right
                r.left = q.left
                r.right = q.right
                q = r
        #最小元素是一个叶结点
        if p is None:
            prev_r.left = None
            self._root = r
            self._root.left = q.left
            self._root.right = q.right
        else:
            prev_r.left = None
            r.left = q.left
            r.right = q.right
            q = r

    def print_tree(self):
        for key, value in self.entries():
            print(key, value)

#构造二叉排序树
def build_dictBinTree(entries):
    dic = DictBinTree()

    for key , value in entries:
        assoc = Assoc(key,value)
        dic.insert(assoc)
    
    return dic

class DictOptBinTree(DictBinTree):
    def __init__(self,seq):
        DictBinTree.__init__(self)
        data = sorted(seq)
        self._root = DictOptBinTree.build_OBT(data,0,len(data)-1)

    @staticmethod #注意这里的静态方法
    def build_OBT(data,start,end):
        if start > end:
            return None
        
        mid = (start + end) / 2
        left = DictOptBinTree.build_OBT(data,0,mid-1)
        right = DictOptBinTree.build_OBT(data,mid+1,len(data)-1)
        return BinTNode(Assoc(data[mid][1],data[mid][2]), left , right)
    
    @staticmethod
    def build_OBT_DP(wp,wq):
        dimension = len(wq) #矩阵维度

        w = [[0] * dimension for i in range(dimension)]
        e = [[0] * dimension for i in range(dimension)]
        r = [[0] * dimension for i in range(dimension)]

        # 构造w矩阵
        for i in range(dimension):
            w[i][i] = wq[i]
            e[i][i] = w[i][i]
            for j in range(i+1, dimension):
                w[i][j] = w[i][j-1] + wp[j-1] + wq[j] #递推式
        
        for i in range(0,dimension-1): #构造已包含一个内部结点的树
            r[i][i+1] = i
            e[i][i+1] = w[i][i+1]

        for m in range(2, dimension):
            #构造包含m个结点的树
            for i in range(0, dimension -m) :
                k_0 , j = 0 , i + m
                wmin = float("inf")
                for k in range(i,j):
                    #找使e[i][k] + e[k+1][j]最小的k
                    if e[i][k] + e[k+1][j] < wmin:
                        wmin = e[i][k] + e[k+1][j]
                        k_0 = k
                
                e[i][j] = w[i][j] + wmin
                r[i][j] = k_0
        
        return e,r
