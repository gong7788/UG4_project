

def decimalToBinary(n):   # converting decimal to binary
    b = 0
    i = 1
    while (n != 0):
        r = n % 2
        b+= r * i
        n//= 2
        i = i * 10
    return b


def makeList(k):       # list of the binary element produced
    a =[]
    if(k == 0):
        a.append(0)
    while (k>0):
        a.append(k % 10)
        k//= 10
    a.reverse()
    return a


def checkBinary(bin, l):
    temp =[]
    for i in range(len(bin)):
        if(bin[i]== 1):
            temp.append(l[i])
    return temp


def get_subsets(l):
    binlist =[]
    subsets =[]
    n = len(l)
    for i in range(2**n):
        s = decimalToBinary(i)
        arr = makeList(s)

        binlist.append(arr)

        for i in binlist:

            k = 0

            while(len(i)!= n):
                i.insert(k, 0) # representing the binary equivalent according to len(l)
                k = k + 1
    for i in binlist:
        subsets.append(checkBinary(i, l))
                                            # print(binlist) print this for more understanding
    return subsets
