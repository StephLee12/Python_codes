# 自己写的 双指针
def v_1(height):
    
    length = len(height)
    max_area = 0
    left_pos = 0
    right_pos = length-1
    left = height[left_pos]
    right = height[right_pos]
    area = 0
    while right_pos - left_pos > 0:      
        if left <= right:
            area = (right_pos-left_pos) * left
            left_pos += 1
            left = height[left_pos]
        else:
            area = (right_pos-left_pos) * right
            right_pos -= 1
            right = height[right_pos]
        if area > max_area:
            max_area = area
    
    return max_area
        