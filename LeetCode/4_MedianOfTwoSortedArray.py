# 纯暴力求解 开了一个新数组 对每一个数组使用一个指针 进行排序 排到新数组里
# 自己写的
def v_1(nums1, nums2):
    median = 0

    nums1_length = len(nums1)
    nums2_pointer, nums2_length = 0, len(nums2)

    new_num = []
    for i in range(nums1_length):
        while nums2_pointer < nums2_length and nums2[nums2_pointer] <= nums1[i]:
            new_num.append(nums2[nums2_pointer])
            nums2_pointer += 1
        new_num.append(nums1[i])

    while nums2_pointer < nums2_length:
        new_num.append(nums2[nums2_pointer])
        nums2_pointer += 1

    if (nums1_length + nums2_length) % 2 == 0:
        median = (float(new_num[int(
            (nums1_length + nums2_length) / 2)]) + float(new_num[int(
                (nums1_length + nums2_length) / 2 - 1)])) / 2
    else:
        median = float(new_num[int((nums1_length + nums2_length - 1) / 2)])

    return median


# 也是暴力 没开数组 只需要
def v_2(nums1, nums2):
    median = 0

    nums1_length = len(nums1)
    nums2_length = len(nums2)
    total_length = nums1_length + nums2_length

    pointer = 0
    nums1_pointer = 0
    nums2_pointer = 0
    if total_length % 2 == 0:  #总长度是偶数
        # left_num_pos,right_num_pos = total_length/2-1,total_length/2
        median_num = [None, None]
        while pointer < (total_length / 2 + 1):

            median_num[0] = median_num[1]
            if nums1_pointer == nums1_length:  # nums1遍历完
                median_num[1] = nums2[nums2_pointer]
                nums2_pointer += 1
                pointer += 1
                continue

            if nums2_pointer == nums2_length:
                median_num[1] = nums1[nums1_pointer]
                nums1_pointer += 1
                pointer += 1
                continue

            if nums1[nums1_pointer] <= nums2[nums2_pointer]:
                median_num[1] = nums1[nums1_pointer]
                nums1_pointer += 1
            elif nums1[nums1_pointer] > nums2[nums2_pointer]:
                median_num[1] = nums2[nums2_pointer]
                nums2_pointer += 1
            pointer += 1
        median = (median_num[0] + median_num[1]) / 2
    else:  #总长度是奇数
        # median_pos = (total_length-1)/2
        while pointer < (total_length + 1) / 2:

            if nums1_pointer == nums1_length:  # nums1遍历完
                median = nums2[nums2_pointer]
                nums2_pointer += 1
                pointer += 1
                continue

            if nums2_pointer == nums2_length:
                median = nums1[nums1_pointer]
                nums1_pointer += 1
                pointer += 1
                continue

            if nums1[nums1_pointer] <= nums2[nums2_pointer]:
                median = nums1[nums1_pointer]
                nums1_pointer += 1
            else:
                median = nums2[nums2_pointer]
                nums2_pointer += 1

            pointer += 1

    return float(median)


# 时间复杂度为 log(m+n) 二分
def v_3(nums1, nums2):
    def kth_min_num(nums1, s_1, d_1, nums2, s_2, d_2, k):
        len_1, len_2 = d_1 - s_1 + 1, d_2 - s_2 + 1

        if len_1 > len_2:
            return kth_min_num(nums2, s_2, d_2, nums1, s_1, d_1, k)
        if len_1 == 0:
            return nums2[s_2 + k - 1]
        if k == 1:
            return min(nums1[s_1], nums2[s_2])

        point_1 = s_1 + min(len_1, int(k / 2)) - 1
        point_2 = s_2 + min(len_2, int(k / 2)) - 1

        if nums1[point_1] <= nums2[point_2]:
            return kth_min_num(nums1, point_1 + 1, d_1, nums2, s_2, d_2,
                               k - (point_1 - s_1 + 1))
        else:
            return kth_min_num(nums1, s_1, d_1, nums2, point_2 + 1, d_2,
                               k - (point_2 - s_2 + 1))

    length_1 = len(nums1)
    length_2 = len(nums2)
    total_length = length_1 + length_2
    # 如果是奇数 left=right
    left = int((total_length + 1) / 2)
    right = int((total_length + 2) / 2)

    return 0.5 * (kth_min_num(nums1, 0, length_1-1, nums2, 0, length_2-1, left) +
                  kth_min_num(nums1, 0, length_1-1, nums2, 0, length_2-1, right))


if __name__ == "__main__":
    median = v_2([1, 2], [3, 4])
