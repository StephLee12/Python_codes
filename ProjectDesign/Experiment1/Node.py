# information gain DT
class IGNode:  
    def __init__(self, data=None, split_att=None, belong_att=None):
        self.data = data
        self.split_att = split_att
        self.belong_att = belong_att
        self.children = []

# inherit ignode
class CARTNode(IGNode):  
    def __init__(self,
                 data=None,
                 split_att=None,
                 belong_att=None,
                 left=None,
                 right=None):
        super(CARTNode, self).__init__(data, split_att, belong_att)
        self.left = left
        self.right = right
