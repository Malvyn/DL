bs = "c上元高会集群仙.".encode('gb2312')
#
for b in bs:
    print("{} ".format(b), end='')


def chinese_to_index(text):
    result = []
    bs = text.encode('gb2312')
    num = len(bs)
    i = 0
    while i < num:
        b = bs[i]
        # 如果出现值小于160，那么表示单个字符
        if b <= 160:
            result.append(b)
        else:
            block = b - 160
            if block >= 16:
                # 10~15分区是空的，所以超过的，不需要考虑
                block -= 6
            # 计算之前有多少数量，所以减上1
            block -= 1
            # 计算位码
            i += 1
            b2 = bs[i] - 160 - 1
            result.append(161 + block * 94 + b2)
        i += 1
    return result

result = chinese_to_index("伍")
print(result)
