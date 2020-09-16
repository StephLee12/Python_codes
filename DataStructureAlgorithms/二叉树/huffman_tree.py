from priqueue import PriQueue
from bintreenode import BinTNode

class HTNode(BinTNode):
    def __lt__(self,othernode):
        return self.data < othernode.data
    
class HuffmanPriQueue(PriQueue):
    def bintree_num(self):
        return len(self._elems)

def HuffmanTree(weights):
    trees = HuffmanPriQueue()
    for w in weights:
        trees.enqueue(HTNode(w))
    while trees.bintree_num() > 1:
        node_left = trees.dequeue()
        node_right = trees.dequeue()
        data_root = node_left.data + node_right.data
        trees.enqueue(HTNode(data_root,node_left,node_right))
    
    return trees.dequeue()