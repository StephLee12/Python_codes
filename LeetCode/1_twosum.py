

# 先排序再遍历  O(n)=nlogn+n
def twoSum(nums,target):
    # nums: list[int]
    # target: int 
    # return type: List[int]

    # 对nums进行排序 返回的是元素对应的下标
    sort_ids = sorted(range(len(nums)),lambda x: nums[x]) 
    idx_1,idx_2 = 0,len(nums)-1
    # 二重遍历 里面是找到第idx_1大和第idx_2大的元素在nums中的下标
    res_sum = nums[sort_ids[idx_1]] + nums[sort_ids[idx_2]] 
    while res_sum != target:
        if res_sum < target:
            idx_1 += 1
        elif res_sum > target:
            idx_2 -= 1
        res_sum = nums[sort_ids[idx_1]] + nums[sort_ids[idx_2]] 
    
    return [sort_ids[idx_1],sort_ids[idx_2]]

# 暴力 直接两个循环
def twoSum_v2(nums,target):

    for i in range(len(nums)-1):
        for j in range(i+1,len(nums)):
            if nums[i] + nums[j] == target:
                return [i,j]

    return None


# 用字典
def twoSum_v3(nums,target):

    num_dict = {}

    for idx,val in enumerate(nums):
        left_val = target - val
        if left_val in num_dict:
            return [num_dict[left_val],idx]
        num_dict[val] = idx
    
    return None
        