import matplotlib.pyplot as plt
import numpy as np
import statistics
import os

def getmedian(ls):
    mediannum = len(ls)//2
    return (ls[mediannum] + ls[~mediannum]) /2
def train2dic(filename):
    with open("./data/"+filename+"/train.txt", "r") as f:
        lines = f.readlines()
        ent_item = []
        du_dic = {}
        for line in lines:
            temp = line.strip().split("\t")
            for item in temp[1:]:
                if du_dic.get(item) == None:
                    du_dic[item] = 1
                elif du_dic.get(item) != None:
                    du_dic[item] += 1
                
    return du_dic

filelist = os.listdir("./data")
print(filelist)
datadic = {}
for file in filelist:
    filedic = train2dic(file)
    ls = [x[1] for x in filedic.items()]
    sortls = sorted(ls)
    # plt.plot([i for i in range(len(sortls))], sortls)
    # plt.show()
    datamean = np.mean(sortls)
    print("--------")
    print(len(sortls))
    print(len([y for y in sortls if y > datamean]))
    datamedian = getmedian(sortls)
    print(len([y for y in sortls if y > datamedian]))
    mean_median_json = {
        "mean": datamean,
        "median": datamedian
    }
    datadic[file] = mean_median_json

print(datadic)
