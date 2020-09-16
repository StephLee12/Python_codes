class StackUnderflow(ValueError):
    pass

class SStack(): #基于顺序表技术实现的栈类

    def __init__(self): #用list对象 _elems存储栈中元素
        self._elems = [] #所有栈操作都映射到list操作
    
    def is_empty(self):
        return self._elems == []

    def depth(self):
        return len(self._elems)
    
    def top(self):
        if self._elems == []:
            raise StackUnderflow("in SStack.top()")

        return self._elems[-1]
    
    def push(self,elem):
        self._elems.append(elem)
    
    def pop(self):

        if self._elems == []:
            raise StackUnderflow("in SStack.pop()")
        
        return self._elems.pop()

def check_brackets(text): #括号匹配问题

    brackets = "()[]{}"
    open_brackets = "([{"
    bra_dict = {")":"(",
                "]":"[",
                "}":"{"} # 配对关系的字典

    def gen_brackets(text):
        # 括号生成器 每次调用返回text里下一括号及其位置
        i, text_len = 0 , len(text)
        while True:
            while i < text_len and text[i] not in brackets:
                i = i + 1
            if i >= text_len:
                return
            yield text[i],i
            i = i + 1
    
    st = SStack() #创建栈

    for pr, i in gen_brackets(text): #对text里的各括号和位置迭代
        if pr in open_brackets: #开括号压入栈中
            st.push(pr)
        elif st.pop() != bra_dict[pr]:
            print("Unmatching is found at",i,"for",pr)
            return False
        else:
            continue
        
    print("All brackets are correctly matched")
    return True

