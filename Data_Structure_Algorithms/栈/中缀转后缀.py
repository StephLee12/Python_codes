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

def suf_exp_getValue(exp): #后缀表达式求值核心函数

    operators = "+-*/"
    st = SStack()

    for x in exp:

        if x not in operators: # x是操作数时 将x压入栈中
            st.push(float(x))
            continue
        #若x是运算符
        if st.depth() < 2: #栈的深度不够时，报错
            raise SyntaxError("Shortage of operand")

        operand_1 = st.pop()
        operand_2 = st.pop()

        if x == "+":
            result = operand_2 + operand_1
        elif x == "-":
            result = operand_2 - operand_1
        elif x == "*":
            result = operand_2 * operand_1
        elif x == "/":#注意这里可能引发除零异常
            if operand_1 == 0:
                raise ZeroDivisionError("One operand is zero")
            else:
                result = operand_2 / operand_1 
        else:
            print("Unknown Error!")
            break
        
        st.push(result)

        if st.depth() == 1:
            return st.pop()
        
        raise SyntaxError("Extra operand")

def suf_exp_splitExp(line): #将输入的字符串分割为项的表
    return suf_exp_getValue(line.split())

def suf_exp_calculator():
    while True:
        try:
            line = input("Suffix Expression: ")
            if line == "end":
                return 
            result = suf_exp_splitExp(line)
            print(result)
            
        except Exception as ex:
            print("Error: ", type(ex),ex.args)

priority = {"(":1,
            "+":3,
            "-":3,
            "*":5,
            "/":5} #字典 代表优先级 开括号优先级最低，保证其他的运算符都不会将其弹出

infix_operators = "+-*/()"

def trans_infix_suffix(line):
    st = SStack()
    suf_exp = []

    for x in generator(line): #tokens是一个生成器
        if x not in infix_operators:# 不是运算对象 直接送入suf_exp
            suf_exp.append(x)
        elif st.is_empty() or x == '(': #左括号进栈
            st.push(x)
        elif x == ')': #遇到右括号时 依次弹出运算符
            while ((not st.is_empty()) and (st.top() != '(')):
                suf_exp.append(st.pop())
            if st.is_empty():
                raise SyntaxError("Missing '(' ")
            st.pop() #将左括号也弹出
        else: #markdown中的第一种情况 比较相邻两个运算符的优先级
            while ((not st.is_empty()) and (priority[st.top()] >= priority[x])):
                suf_exp.append(st.pop())
            st.push(x)
        
        while not st.is_empty(): #送出栈里剩下的运算符
            if st.top() == '(':
                raise SyntaxError("Extra '(' ")
            
            suf_exp.append(st.pop())
        
        return suf_exp

def generator(line):
    #生成器函数 逐一生成line中的每一个项 
    #项是浮点数或是运算符，但是此函数不能处理一元运算符，也不能处理带符号的浮点数
    i , length  = 0 , len(line)

    while i < length:
        while line[i].isspace(): #检测line[i]是否是空格
            i = i + 1
        
        if i >= length:
            break
        
        if line[i] in infix_operators: #如果line[i]是运算符
            yield line[i]
            i = i + 1
            continue

        j = i + 1

        while ((j < length) and (not line[j].isspace() ) and (line[j] not in infix_operators)):
            if (line[j] == 'e' or line [j] == 'E') and (j + 1< length and line[j+1] == '-'):
                j = j + 1

            j = j + 1 
        
        yield line[i:j]
        i = j

def test_trans_infix_suffix():
    while True:
        try:
            line = input("Infix Expression: ")
            if line == "end":
                return 
            print(trans_infix_suffix(line))
            print("Suffix Expression's Value: ", suf_exp_getValue(trans_infix_suffix(line))) 
        except Exception as ex:
            print("Error: ", type(ex),ex.args)