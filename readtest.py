ls = []
with open("./data/FB15k/entities.dict") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        ls.append(line[1])
with open("./data/FB15k/relations.dict") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        ls.append(line[1])
all_ls = []
with open("./data/FB15k/train.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        all_ls = all_ls + line

with open("./data/FB15k/valid.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        all_ls = all_ls + line

with open("./data/FB15k/test.txt") as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        all_ls = all_ls + line
print(len(set(ls)))
print(len(set(all_ls)))