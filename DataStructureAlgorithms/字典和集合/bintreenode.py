class BinTNode():
    def __init__(self,data,left=None,right=None):
        self.data = data
        self.left = left
        self.right = right
    
    def is_empty(self):
        return self.data is None

t = BinTNode(1, BinTNode(2), BinTNode(3))
