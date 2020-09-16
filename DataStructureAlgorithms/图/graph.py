from stack import SStack
from queue import SQueue
from priqueue import PriQueue
class GraphError(ValueError):
    pass

class Graph:
    def __init__(self, mat=[],unconn=0):
        ver_num = len(mat)
        for x in mat:
            if len(x) != ver_num:
                raise ValueError("Matrix is not square matrix")
        
        self._mat = [mat[i][:] for i in range(ver_num)] #通过生成式来生成表
        self._unconn = unconn
        self._ver_num = ver_num
    
    def get_ver_num(self):
        return self._ver_num
    
    def index_invalid(self,index):
        return index < 0 or index >= self._ver_num

    def add_edge(self,vi,vj,val=1):
        if self.index_invalid(vi) or self.index_invalid(vj):
            raise GraphError(str(vi) + 'or' + str(vj) + "is not a valid index")
        elif self._mat[vi][vj] != self._unconn:
            print("modify the value of edge"+ str(vi)+' '+str(vj))
            self._mat[vi][vj] = val
        else:
            print("Create a new edge between vertices" + str(vi) + "and" + str(vj))
            self._mat[vi][vj] = val

    def get_edge(self,vi,vj):
        if self.index_invalid(vi) or self.index_invalid(vj):
            raise GraphError(str(vi) + 'or' + str(vj) + "is not a valid index")
        
        return self._mat[vi][vj]
    
    def get_outedges(self,vi): #n^2的复杂度 有点蛋疼
        if self.index_invalid(vi):
            raise GraphError(str(vi) + "is not a valid index")
        return self._outedges(self._mat[vi],self._unconn)

    @staticmethod
    def _outedges(row,unconn):
        edges = []
        for i in range(len(row)):
            if row[i] != unconn:
                edges.append((i,row[i])) # append是一个元组 第一个elem是顶点的index 第二个elem是权重
        
        return edges
        
class GraphAL:
    def __init__(self,mat=[],unconn=0):
        ver_num = len(mat)
        for x in mat:
            if len(x) != ver_num:
                raise ValueError("Matrix is not square matrix")
        self._list = [Graph._outedges(mat[i],unconn) for i in range(ver_num)]
        self._ver_num = ver_num
        self._unconn = unconn
    
    def get_ver_num(self):
        return self._ver_num
    
    def add_vertex(self):
        self._list.append([])
        self._ver_num = self._ver_num + 1
    
    def index_invalid(self,index):
        return index < 0 or index >= self._ver_num

    def add_edge(self,vi,vj,val=1):
        if self._ver_num() == 0:
            raise GraphError("Cannot add an edge to an empty graph")
        elif self.index_invalid(vi) or self.index_invalid(vj):
            raise GraphError(str(vi) + 'or' + str(vj) + "is not a valid index")
        for i in range(len(self._list[vi])):
            if self._list[vi][i][0] == vj: # self._list[vi][i]是一个元组
                print("modify the value of edge"+ str(vi)+' '+str(vj))
                self._list[vi][i][1] = val
            elif self._list[vi][i][0] > vj: # 如果某一个顶点的index比vj还大 说明vi和vj之间没有边
                break
        print("Create a new edge between vertices" + str(vi) + "and" + str(vj))
        self._list[vi].insert(i,(vj,val))
    
    def get_edge(self,vi,vj):
        if self._ver_num() == 0:
            raise GraphError("Cannot get an edge to an empty graph")
        elif self.index_invalid(vi) or self.index_invalid(vj):
            raise GraphError(str(vi) + 'or' + str(vj) + "is not a valid index")
        else:
            for i in range(len(self._list[vi])):
                if self._list[vi][i][0] == vj:
                    print("Find edge with its weight ",self._list[vi][i][1])
                    return self._list[vi][i][1]
            raise GraphError("Edge doesn't exist")
    
    def get_outedges(self,vi):
        if self.index_invalid(vi):
            raise GraphError(str(vi) + "is not a valid index")
        return self._list[vi]

def DFS_graph_non_recursive(graph,v0): # 借用辅助栈 非递归实现
    ver_num = graph.get_ver_num()
    visited = [0] * ver_num
    visited[v0] = 1
    dfs_seq = [v0] #dfs序列
    st = SStack()
    st.push((0,graph.get_outedges(v0)))
    while not st.is_empty():
        i , edges = st.pop()
        if i < len(edges):
            v, w = edges[i]
            st.push((i+1, edges)) #回溯时访问i+1
            if not visited[v]: #v未被访问，记录并继续dfs
                dfs_seq.append(v)
                visited[v] = 1
                st.push((0,graph.get_outedges(v)))
    return dfs_seq

def DFS_graph_recursive(graph,v,visited,dfs_seq): 
   # 最初输入的visited是一个全0的列表 dfs_seq初始输入为[]
    dfs_seq.append(v) #将v添加的dfs序列
    visited[v] = 1 # 标记v顶点
    v_outedges = graph.get_outedges(v) # 获取出度 是一个元组
    for edge in v_outedges:
        v_next_index = edge[0] 
        if visited[v_next_index] != 1: # 没有被访问过
            DFS_graph_recursive(graph,v_next_index,visited,dfs_seq)
    
    return dfs_seq

def BFS_graph(graph,v0):
    vertex_num = graph.get_ver_num()
    bfs_seq = [v0]
    visited = [0] * vertex_num
    visited[v0] = 1

    qu = SQueue()
    qu.enqueue(graph.get_outedges(v0))

    while not qu.is_empty():
        edges = qu.dequeue()

        for i in range(len(edges)):
            v = edges[i][0]
            if not visited[v]:
                qu.enqueue(graph.get_outedges(v))
                visited[v] = 1
                bfs_seq.append(v)
    
    return bfs_seq

def get_missing_vertex(graph,dfs_seq): #获取未访问的顶点列表
    miss_list = []
    full_list = [i for i in range(graph.get_ver_num())]
    for elem in full_list:
        if elem not in dfs_seq:
            miss_list.append(elem)
    
    return miss_list

def has_traversed(graph,dfs_seq): #判断图是否完全遍历

    if len(dfs_seq) != graph.get_ver_num():
        return False
    else:
        return True

def dfs_continue_traverse(graph,miss_list): #若图未遍历完全 继续遍历

    dfs_bucket = []
    for i in range(len(miss_list)):
        # 以新顶点遍历图 用递归或非递归都可
        dfs_seq = DFS_graph_recursive(graph,miss_list[i],[0]*graph.get_ver_num(),[]) 
        dfs_bucket.append(dfs_seq)
        # 在原miss_list弹出该顶点
        miss_list.pop(i)
        # 获取新的miss_list
        new_miss_list = get_missing_vertex(graph,dfs_seq)
        # 若原miss_list中的元素在新miss_list有出现，要弹出
        for j in range(len(miss_list)):
            if miss_list[j] in new_miss_list:
                miss_list.pop(j)
        if miss_list == []:
            break
    return dfs_bucket

# 有点懒 就把dfs的复制一下 改个变量名
def bfs_continue_traverse(graph,miss_list): #若图未遍历完全 继续遍历

    bfs_bucket = []
    for i in range(len(miss_list)):
        # 以新顶点遍历图
        bfs_seq = BFS_graph(graph,miss_list[i])
        bfs_bucket.append(bfs_seq)
        # 在原miss_list弹出该顶点
        miss_list.pop(i)
        # 获取新的miss_list
        new_miss_list = get_missing_vertex(graph,bfs_seq)
        # 若原miss_list中的元素在新miss_list有出现，要弹出
        for j in range(len(miss_list)):
            if miss_list[j] in new_miss_list:
                miss_list.pop(j)
        if miss_list == []:
            break
    return bfs_bucket


def DFS_SpanTree(graph,v0):
    vertex_num = graph.get_ver_num()
    dfs_spantree = []
    visited  = [0] * vertex_num
    visited[v0] = 1

    st = SStack()
    st.push((0,v0,graph.get_outedges(v0)))


    while not st.is_empty():
        i, prev_v,edges = st.pop()

        if i < len(edges):
            v, w = edges[i]
            st.push((i+1, prev_v,edges))
            if not visited[v]:
                visited[v] = 1 
                st.push((0,v,graph.get_outedges(v)))
                dfs_spantree.append((prev_v,w,v))

    return dfs_spantree

def BFS_SpanTree(graph,v0):
    vertex_num = graph.get_ver_num()
    bfs_spantree = []
    visited = [0] * vertex_num
    visited[v0] = 1

    qu = SQueue()
    qu.enqueue((v0,graph.get_outedges(v0)))

    while not qu.is_empty():
        prev_v , edges = qu.dequeue()
        for i in range(len(edges)):
            v, w = edges[i]
            if not visited[v]:
                visited[v] = 1
                qu.enqueue((v0,graph.get_outedges(v)))
                bfs_spantree.append(prev_v,w,v)
    
    return bfs_spantree

def Kruskal_MST(graph): #选择权重最小的连通分量相连
    vertex_num = graph.get_ver_num()
    reps = [i for i in range(vertex_num)] #代表元
    mst,edges=[],[]
    
    for vi in range(vertex_num):
        for v , w in graph.get_outedges(vi):
            edges.append((w,vi,v)) 

    edges.sort() #边按权重排序

    for w, vi ,vj  in edges :
        if reps[vi] != reps[vj]:
            mst.append((vi,vj),w)
            
            if len(mst) == vertex_num -1 :
                break
            
            for i in range(vertex_num):
                if reps[i] == reps[vj]:
                    reps[i] = reps[vi] #修改元 合并连通分量
    
    return mst

def Prim_MST(graph,v0):
    vertex_num = graph.get_ver_num()
    mst = [None] * vertex_num

    prio_qu = PriQueue((0,v0,v0))
    count = 0

    while count < vertex_num and not prio_qu.is_empty():
        w, prev_v ,v = prio_qu.dequeue()

        if mst[v] is not None: #判断顶点v是否在已连接的树中
            continue
        #若顶点v不在 已经生成的树中 则prev_v和v构成的边一定在最小生成树上 
        mst[v] = ((prev_v,v),w)
        count = count + 1

        for vertex, weight in graph.get_outedges(v):
            if mst[vertex] is None:
                prio_qu.enqueue((weight,prev_v,vertex)) #优先队列入列后会排序
    
    return mst

def Dijkstra(graph,v0):
    vertex_num = graph.get_ver_num()
    min_path = [None] * vertex_num
    
    cands = PriQueue((0,v0,v0))
    count = 0

    while count < vertex_num and not cands.is_empty():
        w, prev_v, v = cands.dequeue()

        if min_path[v] is not None:
            continue
        min_path[v] = (prev_v,w)
        count = count + 1

        for vertex , weight in graph.get_outedges(v):
            if min_path[vertex] is None:
                cands.enqueue((weight,prev_v,vertex))
    
    return min_path

def Floyd(graph):
    vertex_num = graph.get_ver_num()

    a = graph._mat

    inf = float("inf")

    #for i in range(vertex_num):
    #   for j in range(vertex_num):
    #       if a[i][j] == inf:
    #           next_v[i][j] = -1
    #       else:
    #           next_v[i][j] = j
    next_v = [ [-1 if a[i][j]==inf else j for j in range(vertex_num)] for i in range(vertex_num)]

    for k in range(vertex_num):
        for i in range(vertex_num):
            for j in range(vertex_num):
                if a[i][j] > a[i][k] + a[k][j]:
                    a[i][j] = a[i][k] + a[k][j]
                    next_v[i][j] = next_v[i][k]
    
    return (a,next_v)

def topo_sort(graph):
    vertex_num = graph.get_ver_num()
    indegree, topo_seq = [0] * vertex_num , []

    for vi in range(vertex_num):
        for v ,w in graph.get_outedges(vi):
            indegree[v] = indegree[v] + 1
    
    qu = SQueue()
    for vi in range(vertex_num):
        if indegree[vi] == 0:
            qu.enqueue(vi)
    
    if qu.is_empty():
        return False

    while not qu.is_empty():

        v_non_in = qu.dequeue()
        topo_seq.append(v_non_in)

        for v, w in graph.get_outedges(v_non_in):
            indegree[v] = indegree[v] -1
            if indegree[v] == 0:
                qu.enqueue(v)
    
    return topo_seq

def Aoe(graph):
    def ee_time(graph,topo_seq):
        ee = [0] * graph.get_ver_num()

        for i in topo_seq:
            for j,w in graph.get_outedges(i):
                if ee[i] + w > ee[j]:
                    ee[j] = ee[i] + w

        return ee       

    def le_time(graph,topo_seq,eelast):
        le = [eelast] * graph.get_ver_num()

        for k in range(graph.get_ver_num()-2 , -1 ,-1 ): #逆拓扑排序
            i = topo_seq[k]
            for j , w in graph.get_outedges(i):
                if le[j] - w < le[i]:
                    le[i] = le[j] - w    
        
        return le
    
    def crucial_path(ee,le,graph):
        path = []

        for i in range(graph.get_ver_num()):
            for j , w in graph._outedges(i):
                if le[j] - w == ee[i]:
                    path.append((i,j,ee[i]))
        
        return path

    topo_seq = topo_sort(graph)
    if topo_seq is False:
        return False
    
    ee = ee_time(graph,topo_seq)
    le = le_time(graph,topo_seq,ee[-1])
    
    return crucial_path(ee,le,graph)