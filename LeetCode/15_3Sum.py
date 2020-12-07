# 想的是 通过中间的0向两边扩散 太麻烦了 还得考虑有没有0的情况
def v_1(nums):
    length = len(nums)
    if length < 3:
        return []

    triplets = []

    sort_index = sorted(range(length), key=lambda x: nums[x], reverse=True)
    max_positive = nums[sort_index[0]]
    min_negative = nums[sort_index[length - 1]]
    if min_negative == 0 and max_positive == 0:  # 边界情况
        return [[0, 0, 0]]
    if min_negative >= 0 or max_positive <= 0:  # 边界情况
        return []

    zero_pos = []  # 防止有多个0
    first_zero_pos = None  # 距离正数最近的0的下标
    last_zero_pos = None  # 距离负数最近的0的下标
    first_neg_pos = None  # 最大的负数的下标
    last_pos_pos = None  # 最小的正数的下标
    for i in range(length):
        if nums[sort_index[i]] < 0:
            first_neg_pos = i
            break
        if nums[sort_index[i]] == 0:
            zero_pos.append(i)
            continue
        if nums[sort_index[i + 1]] <= 0:
            last_pos_pos = i

    zero_length = len(zero_pos)
    if zero_length == 0:
        first_zero_pos = first_neg_pos
        last_zero_pos = last_pos_pos
    else:
        first_zero_pos = zero_pos[0]
        last_zero_pos = zero_pos[zero_length - 1]

    min_positive = nums[sort_index[first_zero_pos - 1]]
    max_negative = nums[sort_index[last_zero_pos + 1]]

    pos_counter = 1
    neg_counter = 1
    while first_zero_pos - pos_counter >= 0 and last_zero_pos + neg_counter <= length - 1:

        pos_elem = nums[sort_index[first_zero_pos - pos_counter]]
        neg_elem = nums[sort_index[last_zero_pos + neg_counter]]
        two_sum = pos_elem + neg_elem
        if triplets != []:
            if triplets[-1][1] == 0:
                # elem_p,elem_n = triplets[-1][0],triplets[-1][2]
                for i in range(first_zero_pos - pos_counter + 1, last_pos_pos):
                    for j in range(i + 1, last_pos_pos + 1):
                        if [
                                nums[sort_index[i]], nums[sort_index[j]],
                                neg_elem
                        ] != triplets[-1] and nums[sort_index[i]] + nums[
                                sort_index[j]] == abs(neg_elem):
                            triplets.append([
                                nums[sort_index[i]], nums[sort_index[j]],
                                neg_elem
                            ])
                for i in range(last_zero_pos + neg_counter - 1, first_neg_pos,
                               -1):
                    for j in range(i - 1, first_neg_pos - 1, -1):
                        if [
                                pos_elem, nums[sort_index[j]],
                                nums[sort_index[i]]
                        ] != triplets[-1] and abs(
                                nums[sort_index[i]] +
                                nums[sort_index[j]]) == pos_elem:
                            triplets.append([
                                pos_elem, nums[sort_index[j]],
                                nums[sort_index[i]]
                            ])

        if two_sum == 0:
            if zero_length != 0:
                if triplets == []:
                    triplets.append([pos_elem, 0, neg_elem])
                elif [pos_elem, 0, neg_elem] != triplets[-1]:
                    triplets.append([pos_elem, 0, neg_elem])
            pos_counter += 1
            neg_counter += 1
        elif two_sum > 0:
            if two_sum < abs(max_negative):
                neg_counter += 1
            elif abs(two_sum) == max_negative and neg_counter != 1:
                triplets.append([pos_elem, max_negative, neg_elem])
                neg_counter += 1
            else:  # two_sum > abs(max_negative)
                if two_sum <= abs(neg_elem):
                    for i in range(1, neg_counter):
                        elem = nums[sort_index[last_zero_pos + neg_counter -
                                               i]]
                        if elem + two_sum == 0:
                            triplets.append([pos_elem, elem, neg_elem])
                            break
                neg_counter += 1
        else:  # pos_elem < neg_elem => two_sum < 0
            if abs(two_sum) < min_positive:
                pos_counter += 1
            elif abs(two_sum) == min_positive and pos_counter != 1:
                triplets.append([pos_elem, min_positive, neg_elem])
                pos_counter += 1
            else:  # abs(two_sum) > min_positive
                if abs(two_sum) <= pos_elem:
                    for i in range(1, pos_counter):
                        elem = nums[sort_index[first_zero_pos - pos_counter +
                                               i]]
                        if elem + two_sum == 0:
                            triplets.append([pos_elem, elem, neg_elem])
                            break
                pos_counter += 1

        if first_zero_pos - pos_counter < 0 and last_zero_pos + neg_counter >= length:
            break

        if first_zero_pos - pos_counter < 0 and abs(neg_elem) > pos_elem:
            break
        elif first_zero_pos - pos_counter < 0 and abs(neg_elem) <= pos_elem:
            pos_counter -= 1
            continue

        if last_zero_pos + neg_counter >= length and pos_elem > abs(neg_elem):
            break
        elif last_zero_pos + neg_counter >= length and pos_elem <= abs(
                neg_elem):
            neg_counter -= 1
            continue
    return triplets


# 看题解
def v_2(nums):

    length = len(nums)
    if nums == [] or length < 3:
        return []

    triplets = []
    nums = list(sorted(nums))
    for i in range(length):
        if nums[i] > 0:
            return triplets
        if i > 0 and nums[i] == nums[i - 1]:  # 没想到的地方
            continue
        left_pointer = i + 1
        right_pointer = length - 1

        while left_pointer < right_pointer:
            if nums[i] + nums[left_pointer] + nums[right_pointer] == 0:
                triplets.append(
                    [nums[i], nums[left_pointer], nums[right_pointer]])
                while left_pointer < right_pointer and nums[
                        left_pointer] == nums[left_pointer + 1]:
                    left_pointer += 1  # 没想到的地方
                while left_pointer < right_pointer and nums[
                        right_pointer] == nums[right_pointer - 1]:
                    right_pointer -= 1  # 没想到的地方
                left_pointer += 1
                right_pointer -= 1
            elif nums[i] + nums[left_pointer] + nums[right_pointer] < 0:
                left_pointer += 1
            else:
                right_pointer -= 1

    return triplets


if __name__ == "__main__":
    v_1([-1, 0, 1, 2, -1, -4, -2, -3, 3, 0, 4])
