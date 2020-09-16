from assoc import Assoc

class DictList:
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def get_length(self):
        return len(self._elems)
    
    def add_elem(self,Assoc):
        self._elems.append(Assoc)
    
    def pop_elem(self,key):
        for i in range(self.get_length()):
            if self._elems[i].key == key:
                return self._elems.pop(i).value
    
    def search(self,key):
        for assoc_obj in self._elems:
            if assoc_obj.key == key:
                return assoc_obj.value

class DictOrdList:
    def __init__(self):
        self._elems = []
    
    def is_empty(self):
        return self._elems == []
    
    def get_length(self):
        return len(self._elems)
    
    def add_elem(self,Assoc):
        key ,value = Assoc.key, Assoc.value
        for i in range(self.get_length()):
            if self._elems[i].key == key:
                self._elems.insert(i,value)
                break
    
    def pop_elem(self,key):
        for i in range(self.get_length()):
            if self._elems[i].key == key:
                return self._elems.pop(i).value
    
    def search(self,key):
        left , right = 0, self.get_length() -1 

        while left <= right:
            mid = (left + right) /2
            if self._elems[mid].key == key: #正好中间位置的key与要检索的key相同
                return self._elems[mid].value
            elif self._elems[mid].key < key: #要检索的key在后半区间
                left = mid + 1
            else: #要检索的区间在前半区间
                right = mid - 1
                