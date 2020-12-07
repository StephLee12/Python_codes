# 自己写 AC
def v_1(digits):
    # Type digits=>str
    length = len(digits)
    res = []
    
    if length == 0:
        return []
    
    phone_dict = {
        '2':['a','b','c'],
        '3':['d','e','f'],
        '4':['g','h','i'],
        '5':['j','k','l'],
        '6':['m','n','o'],
        '7':['p','q','r','s'],
        '8':['t','u','v'],
        '9':['w','x','y','z']
    }
    
    res_dict = {}
    if length >= 2:
        for letter in phone_dict[digits[0]]:
            res_dict[letter] = phone_dict[digits[1]]

        for i in range(2,length):
            letters = phone_dict[digits[i]]
            for key,val_list in res_dict.items():
                new_list = []
                for val in val_list:
                    for letter in letters:
                        new_list.append(val+letter)
                res_dict[key] = new_list
    
        for key,val_list in res_dict.items():
            for val in val_list:
                res.append(key+val)
    else:
        for letter in phone_dict[digits[0]]:
            res_dict[letter] = None
        for key,_ in res_dict.items():
            res.append(key)

    return res

# 看题解 递归(回溯)
def v_2(digits):
    length = len(digits)
    res = []
    
    if length == 0:
        return []
    
    phone_dict = {
        '2':['a','b','c'],
        '3':['d','e','f'],
        '4':['g','h','i'],
        '5':['j','k','l'],
        '6':['m','n','o'],
        '7':['p','q','r','s'],
        '8':['t','u','v'],
        '9':['w','x','y','z']
    }

    def backtrack(combination,nextdigit):
        if len(nextdigit) == 0:
            res.append(combination)
        else:
            for letter in phone_dict[nextdigit[0]]:
                backtrack(combination+letter,nextdigit[1:])
    
    backtrack('',digits)
    return res


if __name__ == "__main__":
    v_1('2')
