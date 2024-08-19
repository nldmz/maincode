with open("./data/FB-AUTO/train.txt","r",encoding="utf-8") as f:
    temp_dic = {}

    lines = f.readlines()

    for line in lines:
        a = line.strip().split("\t")
        r = a[0]
        if r not in temp_dic:
            temp_dic[r] = 1
        else:
            temp_dic[r] += 1

    
print(temp_dic)