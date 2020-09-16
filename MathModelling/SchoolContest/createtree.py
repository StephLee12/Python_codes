from treelib import Tree ,Node
tree = Tree()
tree.create_node(None,"root")
data =[]
num=0 #树的结点个数
for line in open("Modelling/data.txt",'r',encoding='utf-8'):
    num=num+1
    line=line[:-1] #去掉末尾换行符
    data.append(line)
renamelist = [] #用来存储由于重名但不同品牌的零件更改后的id
afterwardlist=[]#用来存储组合名的后一个
forwardlist=[]#存储前一个组合名

for i in range(num):
    s=data[i].split(" ")
    count= len(s)  #按照厂家的分类方式 count = 5
   # print('i='+ str(i))
   # print("count = "+str(count))
    for j in range(count):
        if s[j]=='0':
            tree.remove_node(s[j-1])
        if j == 0: #第一个是厂家
            if tree.contains(s[0]): #树里已经有厂家了
                continue  #跳过直接下一个
            else:
                tree.create_node(s[0],s[0],parent='root')
        if j > 0:
            if j == count-1 :
                continue
            if tree.contains(s[j]): #有这个节点
                if tree.parent(s[j]).identifier not in afterwardlist and tree.parent(s[j]).identifier not in renamelist:
                    if tree.parent(s[j]).identifier == s[j-1]:
                        continue
                    if tree.parent(s[j]).identifier != s[j - 1] and s[j-1]+s[j] not in renamelist:  # 有这个结点 但是他的父节点不是该字符串的上一个值，这个s【j】是新的结点
                        tree.create_node(s[j - 1] + s[j], s[j - 1] + s[j], parent=s[j - 1], data=s[j + 1])
                        renamelist = renamelist + [s[j - 1] + s[j]]
                        forwardlist = forwardlist + [s[j - 1]]  # 存储前一个组合名
                        afterwardlist = afterwardlist + [s[j]]
                    elif tree.parent(s[j]).identifier != s[j - 1] and s[j-1]+s[j] in renamelist:
                        continue

                if tree.parent(s[j]).identifier in afterwardlist or tree.parent(s[j]).identifier in renamelist :
                    tree.create_node(s[j-2]+s[j], s[j-2]+s[j], s[j - 2] + s[j - 1], s[j + 1])
                    renamelist = renamelist + [s[j - 2] + s[j]]
                    forwardlist = forwardlist + [s[j - 2]]  # 存储前一个组合名
                    afterwardlist = afterwardlist + [s[j]]
            else:#没有这个结点
                if s[j-1] in afterwardlist:
                    if tree.contains(s[j-2]+s[j-1]):
                        tree.create_node(s[j],s[j],s[j-2]+s[j-1],s[j+1])
                    elif tree.contains(s[j - 3] + s[j - 1]):
                        tree.create_node(s[j], s[j], s[j - 3] + s[j - 1], s[j + 1])
                    else:
                        tree.create_node(s[j], s[j], s[j - 1], s[j + 1])
                else:
                    tree.create_node(s[j],s[j],s[j-1],s[j+1])
tree.show()