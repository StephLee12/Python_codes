dirs = [(0,1),(1,0),(0,-1),(-1,0)]

def mark(maze,pos): #标记搜索过的位置为2
    maze[pos[0]][pos[1]] == 2

def passable(maze,pos): #判断位置pos是否可以通行
    return maze[pos[0]][pos[1]] == 0

def find_path(maze,pos,end):
    mark(maze,pos)

    if pos == end:
        print(pos, end="")
        return True
    
    for i in range(4):
        next_pos = pos[0] + dirs[i][0], pos[1] + dirs[i][1]

        if passable(maze,next_pos):
            if find_path(maze,next_pos,end):
                print(pos,end="")
                return True
    
    return True