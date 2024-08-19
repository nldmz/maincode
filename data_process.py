import os
import numpy as np
import torch
from collections import defaultdict
from gensim.models import Word2Vec
from itertools import combinations
import math

class Dataset:
    def __init__(self, data_dir, arity_lst, device):
        self.data_dir = data_dir
        self.ent2id, self.rel2id = self.str2id(self.data_dir)
        self.device = device
        self.arity_lst = arity_lst
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.load_data(self.data_dir, arity_lst)
        self.num_ent = len(self.ent2id)
        self.num_rel = len(self.rel2id)
        self.batch_index = 0
        self.num_feature = self.feature_init()
        self.edge_index = self.edge_index_init()
        print(self.num_feature)
        print(self.num_feature.shape)
        print(self.edge_index)
        self.ent2du, self.dumean, self.beyondmeanent, self.weight = self.get_entdu(self.train)
        self.allweight = self.getentweight(self.weight)
        # print(self.dumean)
        # print(len(self.beyondmeanent))
        # print(len(self.allweight))
        self.chosenrelnum = int((self.num_rel - 1)//math.log(self.num_rel))
        self.anchor_num_feature = torch.rand((self.chosenrelnum, 1000), dtype=torch.float32)
        self.relanchor_edge_index = self.anchor_init()
    def anchor_init(self):
        anchorls = [i for i in range(self.chosenrelnum)]
        rells = [i for i in range(self.chosenrelnum, 2*self.chosenrelnum)]
        # print(anchorls, rells)
        allls = [[],[]]
        for i in range(self.chosenrelnum):
            for j in range(self.chosenrelnum):
                allls[1].append(anchorls[i])
                allls[0].append(rells[j])
        for i in range(self.chosenrelnum):
            for j in range(self.chosenrelnum):
                allls[0].append(anchorls[j])
                allls[1].append(rells[i])
        
        # print(allls)
        relanchor_edge_index = torch.tensor(allls, dtype=torch.long).contiguous().to(self.device)
        return relanchor_edge_index
    def getentweight(self, weight):
        allweight = [round(weight, 3) if ele+1 in self.beyondmeanent else 1-round(weight, 3) for ele in range(self.num_ent-1)]
        return allweight
    
    def get_entdu(self, train):
        dictemp = {}
        for k, v in train.items():
            dataforarity = v
            for datas in dataforarity:
                for data in datas[1:]:
                    if dictemp.get(data) is None:
                        dictemp[data] = 1
                    else:
                        dictemp[data] += 1
        du_mean = np.mean([x[1] for x in dictemp.items()])
        beyondmeanent = [x[0] for x in dictemp.items() if x[1] > du_mean]
        weight = 1 - len(beyondmeanent)/self.num_ent
        return dictemp, du_mean, beyondmeanent, weight
    def edge_index_init(self):
        with open(os.path.join(self.data_dir, "train.txt"), "r") as trainf:
            lines = trainf.readlines()
            hyperedges = []
            for line in lines:
                line = line.strip().split("\t")
                rel = self.rel2id[line[0]] - 1
                ents = [self.ent2id[i]+self.num_rel-2  for i in line[1:]]
                hyperedges.append([rel] + ents)
        edges = []

        for row in hyperedges:
            edge_id = row[0]
            entity_ids = row[1:]
            for entity_id in entity_ids:
                edges.append((edge_id, entity_id))
                edges.append((entity_id, edge_id))

        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous().to(self.device)

        edge_index = edge_index[:, edge_index[1].argsort()]
        print(edge_index)
        return edge_index
    def feature_init(self):
        
        num_feature = []
        #加载word2vec模型
        model = Word2Vec.load("./word2vecmodels/" + self.data_dir.split("/")[-2] + "1001/"+"word.model")
        for k,v in self.rel2id.items():
            if v != 0:
                num_feature.append(model.wv[k].tolist())
        for k, v in self.ent2id.items():
            if v != 0:
                num_feature.append(model.wv[k].tolist())
        num_feature = torch.tensor(num_feature).to(self.device)
        return num_feature 

    def load_data(self, data_dir, arity_lst):
        print("Loading {} Dataset".format(data_dir.split("/")[-1]))
        self.train = self.read_tuples(os.path.join(data_dir, "train.txt"), arity_lst, "train")
        self.valid = self.read_tuples(os.path.join(data_dir, "valid.txt"), arity_lst, "valid")
        self.test = self.read_tuples(os.path.join(data_dir, "test.txt"), arity_lst, "test")
        self.all_test = []
        self.all_valid = []
        for arity in arity_lst:
            for fact in self.valid[arity]:
                self.all_valid.append(fact)
            for fact in self.test[arity]:
                self.all_test.append(fact)
    

    def read_tuples(self, dataset, arity_lst, mode):
        if mode == "train":
            self.inc = defaultdict(list)
        data = {}
        for arity in arity_lst:
            data[arity] = []
        with open(dataset) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                rel = self.rel2id[line[0]]
                ents = [self.ent2id[i] for i in line[1:]]
                arity = len(ents)
                if arity in data:
                    data[arity].append(np.array([rel]+ents))
        return data



    def str2id(self, path):
        ent2id, rel2id = {"": 0}, {"": 0}
        with open(os.path.join(path, "entities.dict")) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                id, ent = line[0], line[1]
                if ent not in ent2id:
                    ent2id[ent] = int(id)+1
            f.close()

        with open(os.path.join(path, "relations.dict")) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip().split("\t")
                id, rel = line[0], line[1]
                if rel not in rel2id:
                    rel2id[rel] = int(id)+1
            f.close()

        return ent2id, rel2id

    def next_pos_batch(self, batch_size, arity):
        if self.batch_index + batch_size < len(self.train[arity]):
            batch = self.train[arity][self.batch_index: self.batch_index + batch_size]
            self.batch_index += batch_size
        elif self.batch_index + batch_size >= len(self.train[arity]):
            batch = self.train[arity][self.batch_index:]
            self.batch_index = 0
        batch = np.append(batch, -np.ones((len(batch), 1)), axis=1).astype("int")  # appending the +1 label
        batch = np.append(batch, np.zeros((len(batch), 1)), axis=1).astype("int")  # appending the 0 arity
        return batch

    def next_batch(self, batch_size, neg_ratio, arity, device):
        pos_batch = self.next_pos_batch(batch_size, arity)
        batch = self.generate_neg(pos_batch, neg_ratio, arity)
        batch = torch.tensor(batch).long().to(device)

        return batch


    def generate_neg(self, pos_batch, neg_ratio, arity):
        arities = [arity + 1 - (t == 0).sum() for t in pos_batch[:, 1:]]
        pos_batch[:, -1] = arities
        # print(pos_batch)
        neg_batch = np.concatenate([self.neg_each(np.repeat([c], neg_ratio * arities[i] + 1 + arities[i] -1 , axis=0), arities[i], neg_ratio) for i, c in enumerate(pos_batch)], axis=0)
        return neg_batch
    def neg_each(self, arr, arity, nr):
        arr[0,-2] = 1
        elements = np.array([i+1 for i in range(self.num_ent-1)])
        for p in range(arity-1):
            arr[p+1] = self.changeposition(arr[p+1], p+1)
        for a in range(arity):  #if arity == 2 , a is 0~1
            unwanted_ent = arr[0,a+1]
            temp_allweight = np.array(self.allweight)
            # print(len(temp_allweight))
            temp_allweight = temp_allweight[elements != unwanted_ent]
            temp_elements = elements[elements != unwanted_ent]

            arr[a* nr + 1 + arity - 1:(a + 1) * nr + 1 + arity - 1, a + 1] = np.random.choice(temp_elements, size=nr, p=temp_allweight/temp_allweight.sum())  # a == 0, [1:11, 1] a == 1, [11:21, 2]
            # print(arr)
            # print(arr)
        return arr
    
    def changeposition(self, ls,pos):
        ls[pos], ls[pos+1] = ls[pos+1], ls[pos]
        return ls



    def is_last_batch(self):
        return (self.batch_index == 0)

