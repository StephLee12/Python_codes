class NetNode:
    def __init__(self,idx,degree,ks=0,flag_s=1,flag_i=0,flag_r=0):
        self.idx = idx #节点的素银
        self.degree = degree #节点的度

        self.ks = ks # k-shell值

        self.rank = 0 # 概率模型中节点的rank值
        self.score = 0 # 概率模型中节点的score
        self.uninf_s = 0 #概率模型中的uninf_s
        self.visited = 0 # 用于bfs 1表示访问过

        self.flag_s = flag_s # 节点是否为S类 0假1真
        self.flag_i = flag_i # 节点是否为I类 0假1真
        self.flag_r = flag_r # 节点是否为R类 0假1真
        self.infected_record = [] #记录每个步长的被感染的数目
        self.suspect_record = [] #记录每个步长疑似的数目
        self.removed_record = [] #记录每个步长康复的数目
        self.sir_val = [] # 感染到的节点的数目

        