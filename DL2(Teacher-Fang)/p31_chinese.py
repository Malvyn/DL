
for first in range(8*16+1, 15*16+15):  #  81 - FE    7*16+14=126
    for second in range(4*16, 15*16+15):  #  40 - FE   190
        try:
            bs = bytes([first, second])
            ch = bs.decode('GB18030')
            print(ch, end=' ')
        except:
            continue
    print()


s = '中华人民共和国'
bs = s.encode('GB18030')
for b in bs:
    print(b, end=',')
print()

