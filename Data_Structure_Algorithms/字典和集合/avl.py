from bintreenode import BinTNode
from dictbtree import DictBinTree
from assoc import Assoc
from queue import SQueue
from stack import SStack


class AVLNode(BinTNode):
    def __init__(self,data):
        BinTNode.__init__(self,data)
        self.bf = 0

class DictAVL(DictBinTree):
    def __init__(self):
        DictBinTree.__init__(self)

    @staticmethod
    def LL(a,b):
        a.left = b.right
        b.right = a
        a.bf = b.bf = 0

        return b

    @staticmethod
    def RR(a,b):
        a.right = b.left
        b.left = a
        a.bf = b.bf = 0

        return b

    @staticmethod
    def LR(a,b):
        c = b.right
        a.left , b.right = c.right , c.left
        c.left , c.right = b, a #bca

        if c.bf == 0: #c本身就是插入结点
            a.bf = b.bf = 0
        elif c.bf == 1: #新插入的结点在c的左子树
            a.bf = -1
            b.bf = 0
        elif c.bf == -1: #新插入的结点在c的右子树
            a.bf = 0
            b.bf = -1

        c.bf = 0
        return c

    @staticmethod
    def RL(a,b):
        c = b.left
        a.right ,b.left  = c.left , c.right
        c.left , c.right = a,b
        if c.bf ==0:
            a.bf = b.bf = 0
        elif c.bf == 1: #插入c的左子树
            a.bf = 0
            b.bf = -1
        elif c.bf == -1 :# 插入c的右子树
            a.bf = -1
            b.bf = 0
        c.bf = 0
        return c

    @staticmethod
    def del_adjustment_part(pn,ppn,pnlr,flag):
        #flag == -1 <=> pn.bf == -2
        if flag == -1:
            if pnlr.bf == -1:
                b = DictAVL.RR(pn,pnlr)
                if ppn.left == pn:
                    ppn.left = b
                else: #ppn.right == pn
                    ppn.right = b
                return -1
            elif pnlr.bf == 0:
                b = DictAVL.RR(pn,pnlr)
                b.bf = 1
                if ppn.left == pn:
                    ppn.left = b
                else: #ppn.right == pn
                    ppn.right = b
                return 0
            else: #pnlr.bf == 1
                c = DictAVL.RL(pn,pnlr)
                if ppn.left == pn:
                    ppn.left = b
                else: #ppn.right == pn
                    ppn.right = b
                return 1
        
        else: # flag == 1 <=> pn.bf == 2
            if pnlr.bf == -1:  #要继续回溯
                c = DictAVL.LR(pn, pnlr)
                if ppn.left == pn:
                    ppn.left = c
                else:  #ppn.right == pn
                    ppn.right = c
                return -1
            elif pnlr.bf == 0:  #不需要继续回溯 树高不发生变化
                b = DictAVL.LL(pn, pnlr)
                b.bf = -1  #要修改树根的bf
                if ppn.left == pn:
                    ppn.left = b
                else:  #ppn.right == pn
                    ppn.right = b
                return 0
            else:  #pnl.bf == 1 需要继续回溯
                b = DictAVL.LL(pn, pnlr)
                if ppn.left == pn:
                    ppn.left = b
                else:  #ppn.right == pn
                    ppn.right = b
                return 1

    @staticmethod
    def del_adjustment(p,parent_nodes):
        while not parent_nodes.is_empty():
            pn = parent_nodes.pop()
            if pn.left == p: 
                pn.bf = pn.bf - -1
                if pn.bf == -1:
                    return None
                if pn.bf == -2:
                    ppn = parent_nodes.pop()
                    pnr = pn.right
                    flag = DictAVL.del_adjustment_part(pn,ppn,pnr,-1)
                    if flag == 0:
                        return None

            else: #pn.right == p
                pn.bf = pn.bf + 1 #修改bf值
                if pn.bf == 1:
                    return None
                if pn.bf == 2:
                    ppn = parent_nodes.top() #获得pn结点的父结点
                    pnl = pn.left #获得pn结点的左子树
                    flag = DictAVL.del_adjustment_part(pn,ppn,pnl,1)
                    if flag == 0:
                        return None       
            p = pn

    @staticmethod
    def del_adjustment_leaf(root,p,q,parent_nodes):
        if p == root: #树只有一个根结点 直接删除
            root = None
            return None

        if q.right == p: #如果删除的是父结点的右子叶结点
            q.right = None #删除操作
            #开始回溯
            DictAVL.del_adjustment(p,parent_nodes)
            return None

        else: #如果删除的是父结点的左子叶结点
            q.left = None#删除操作
            #开始回溯
            DictAVL.del_adjustment(p,parent_nodes)
            return None

    @staticmethod
    def del_adjustment_ol(p,q,parent_nodes):
        pl = p.left #p的左子树
        p.data.key , p.data.value = pl.data.key, pl.data.value #将p的左子树的assoc与p交换
        p.left = None #删除操作

        if q.right == p: #如果删除的是父结点的右子结点
            DictAVL.del_adjustment(p,parent_nodes)
            return None
        else: #如果删除的是父结点的左子结点
            DictAVL.del_adjustment(p,parent_nodes)
            return None

    @staticmethod
    def del_adjustment_or(p,q,parent_nodes):
        pl = p.right #p的右子树
        p.data.key , p.data.value = pl.data.key, pl.data.value #将p的右子树的assoc与p交换
        p.left = None #删除操作

        if q.right == p: #如果删除的是父结点的右子结点
            DictAVL.del_adjustment(p,parent_nodes)
            return None
        else: #如果删除的是父结点的左子结点
            DictAVL.del_adjustment(p,parent_nodes)
            return None

    def insert(self,key,value):
        a = p = self._root
        if self._root is None:
            self._root = AVLNode(Assoc(key,value))
            return

        pa = q = None

        #q,p是用来搜索的 a是非平衡子树的树根 pa是非平衡子树树根的父结点
        while p is not None:
            if key == p.data.key: #要插入的assoc对象的key已经存在 替换value
                p.data.value = value
                return

            if p.bf != 0:
                pa, a = q, p  #记录非平衡子树

            q = p #递归 q是p的父结点
            if key < p.data.key:
                p = p.left
            else:
                p = p.right

        # 得到插入点的父结点q 最小非平衡子树a 最小非平衡子树的父结点pa
        node = AVLNode(Assoc(key,value))
        #将结点插入
        if key < q.data.key:
            q.left = node #作为左子结点
        else:
            q.right = node #作为右子结点

        if key < a.data.key:
            p = b = a.left #新结点在a的左子树
            d = 1 #d是a左右子树的高度差
        else:
            p = b = a.right #新结点在a的右子树
            d = -1
        #插入之后 要修改bf值
        while p != node:
            if key < p.data.key: #新插入结点在a的左子树 左子树增高
                p.bf = 1
                p = p.left
            else: #新插入的结点在a的右子树 右子树增高
                p.bf = -1
                p = p.right

        if a.bf == 0: #a的原BF为0 不会失衡
            a.bf = d
            return
        if a.bf == -d: #新结点插入在了a的较低的子树
            a.bf = 0
            return

        if d == 1: #新结点在a的左子树
            if b.bf == 1:
                b = DictAVL.LL(a,b) #新结点在a的左子树的左子树
            else:
                b = DictAVL.LR(a,b) #新结点在a的左子树的右子树
        else:
            if b.bf == -1:
                b = DictAVL.RR(a,b) #新结点在a的右子树的右子树
            else:
                b = DictAVL.RL(a,b) #新结点在a的右子树的左子树
        #修改pa与其子树的联系
        if pa is None: #a为树根
            self._root = b
        else:
            if pa.left == a:
                pa.left = b
            else:
                pa.right = b

    def delete(self,key):
        p, q = self._root, None
        parent_nodes = SStack() #记录所有父结点
        #检索 并记录路径
        while p is not None and key != p.data.key: 
            q = p
            if key < p.data.key:
                p = p.left
                parent_nodes.push(q)
            else:
                p = p.right
                parent_nodes.push(q)
        #没有找到删除结点
        if key != p.data.key: 
            return

        assoc = p
        #删除叶结点
        if p.right == None and p.left == None:
            DictAVL.del_adjustment_leaf(self._root,p,q,parent_nodes)
            return assoc             
        
        # 只有左子树
        if p.left is not None and p.right is None:            
            DictAVL.del_adjustment_ol(p,q,parent_nodes)
            return assoc

        # 只有右子树
        if p.left is None and p.right is not None:
            DictAVL.del_adjustment_or(p,q,parent_nodes)
            return assoc
        #左右子树都有
        if p.left is not None and p.right is not None:
            p_traverse,q_traverse = p,None
            if p.bf == -1: #若右子树高 
                #找到右子树的最左结点 一定是一个叶结点
                while p_traverse is not None:
                    q_traverse = p_traverse #q_traverse是p_traverse的父结点
                    p_traverse = p_traverse.right
                    parent_nodes.push(p_traverse)
                
                p.data.key , p.data.value = p_traverse.data.key, p_traverse.data.value #进行交换

                DictAVL.del_adjustment_leaf(self._root,p_traverse,q_traverse,parent_nodes) #进行调整
                return assoc

            elif p.bf == 0 : #两棵子树一样高 左子树最右 右子树最左都可以
                 #找到右子树的最左结点 一定是一个叶结点
                while p_traverse is not None:
                    q_traverse = p_traverse #q_traverse是p_traverse的父结点
                    p_traverse = p_traverse.right
                    parent_nodes.push(p_traverse)
                
                p.data.key , p.data.value = p_traverse.data.key, p_traverse.data.value #进行交换
                DictAVL.del_adjustment_leaf(self._root,p,q,parent_nodes)
                return assoc
            else: #p.bf == 1
                while p_traverse is not None:
                    q_traverse = p_traverse #q_traverse是p_traverse的父结点
                    p_traverse = p_traverse.left
                    parent_nodes.push(p_traverse)
                
                p.data.key , p.data.value = p_traverse.data.key, p_traverse.data.value #进行交换
                DictAVL.del_adjustment_leaf(self._root,p,q,parent_nodes)
                return assoc

