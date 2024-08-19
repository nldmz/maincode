import argparse

import torch
from data_process import Dataset
from model import HyIE
import numpy as np
import math
from torch.nn import functional as F
from tester import Tester
import os
import json
import numpy as np
import random

# np.random.seed(917)
# random.seed(917)
# torch.manual_seed(917)
# torch.cuda.manual_seed_all(917)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.enabled = False
def seed_torch(seed=917):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def save_model(model, opt, measure, args, measure_by_arity=None, test_by_arity=False, itr=0, test_or_valid='test', is_best_model=False):
    """
    Save the model state to the output folder.
    If is_best_model is True, then save the model also as best_model.chkpnt
    """
    if is_best_model:
        torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.chkpnt'))
        print("######## Saving the BEST MODEL")

    model_name = 'model_{}itr.chkpnt'.format(itr)
    opt_name = 'opt_{}itr.chkpnt'.format(itr) if itr else '{}.chkpnt'.format(args.model)
    measure_name = '{}_measure_{}itr.json'.format(test_or_valid, itr) if itr else '{}.json'.format(args.model)
    print("######## Saving the model {}".format(os.path.join(args.output_dir, model_name)))

    torch.save(model.state_dict(), os.path.join(args.output_dir, model_name))
    torch.save(opt.state_dict(), os.path.join(args.output_dir, opt_name))
    if measure is not None:
        measure_dict = vars(measure)
        # If a best model exists
        if is_best_model:
            measure_dict["best_iteration"] = model.best_itr.cpu().item()
            measure_dict["best_mrr"] = model.best_mrr.cpu().item()
        with open(os.path.join(args.output_dir, measure_name), 'w') as f:
            json.dump(measure_dict, f, indent=4, sort_keys=True)
    # Note that measure_by_arity is only computed at test time (not validation)
    if (test_by_arity) and (measure_by_arity is not None):
        H = {}
        measure_by_arity_name = '{}_measure_{}itr_by_arity.json'.format(test_or_valid,
                                                                        itr) if itr else '{}.json'.format(
            args.model)
        for key in measure_by_arity:
            H[key] = vars(measure_by_arity[key])
        with open(os.path.join(args.output_dir, measure_by_arity_name), 'w') as f:
            json.dump(H, f, indent=4, sort_keys=True)


def decompose_predictions(targets, predictions, max_length):
    positive_indices = np.where(targets > 0)[0]
    seq = []
    for ind, val in enumerate(positive_indices):
        if (ind == len(positive_indices) - 1):
            seq.append(padd(predictions[val:], max_length))
        else:
            seq.append(padd(predictions[val:positive_indices[ind + 1]], max_length))
    return seq


def padd(a, max_length):
    b = F.pad(a, (0, max_length - len(a)), 'constant', -math.inf)
    return b


def padd_and_decompose(targets, predictions, max_length):
    seq = decompose_predictions(targets, predictions, max_length)
    return torch.stack(seq)

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def main(args):
    with open("./data/FB-AUTO/train.txt","r") as countf:
        lines = countf.readlines()
        datacount = {}
        for line in lines:
            temp = line.split('\t')
            num = len(temp)
            if num in datacount:
                datacount[num] += 1
            else:
                datacount[num] = 1
    artiy_lst = []
    for k,v in datacount.items():
        artiy_lst.append(k-1)
    print(datacount)
    args.arity_lst = sorted(artiy_lst)
    print(args.arity_lst)
    # args.arity_lst = [2, 3, 4, 5, 6, 7, 8, 9]
    max_arity = args.arity_lst[-1]
    args.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = Dataset(data_dir=args.dataset, arity_lst=args.arity_lst, device=args.device)
    model = HyIE(dataset, args.emb_dim, args.emb_dim1).to(args.device)
    opt = torch.optim.Adagrad(model.parameters(), lr=args.lr)


    par_dict = get_parameter_number(model)
    print(par_dict)
    for name, param in model.named_parameters():
        print('Parameter %s: %s, require_grad = %s' % (name, str(param.size()), str(param.requires_grad)))
    # If the number of iterations is the same as the current iteration, exit.
    if (model.cur_itr.data >= args.num_iterations):
        print("*************")
        print("Number of iterations is the same as that in the pretrained model.")
        print("Nothing left to train. Exiting.")
        print("*************")
        return

    print("Training the {} model...".format(args.model))
    print("Number of training data points: {}".format(dataset.num_ent))

    print("Starting training at iteration ... {}".format(model.cur_itr.data))
    test_by_arity = args.test_by_arity
    best_model = None
    for it in range(model.cur_itr.data, args.num_iterations + 1):

        model.train()
        model.cur_itr.data += 1
        losses = 0
        for arity in args.arity_lst:
            last_batch = False
            while not last_batch:
                batch = dataset.next_batch(args.batch_size, args.nr, arity, args.device)
                targets = batch[:, -2].cpu().numpy()
                labels = torch.tensor(targets).to(args.device)
                batch = batch[:, :-2]
                last_batch = dataset.is_last_batch()
                opt.zero_grad()
                loss = model.forward(batch, labels)

                loss.backward()
                # print(model.HGNN.probembedding.weight.grad)
                opt.step()
                losses += loss.item()

        print("Iteration#: {}, loss: {}".format(it, losses))
        with open("./results/validresult/trainresult.txt", "a") as f:
            f.write("Iteration#: {}, loss: {}\n".format(it, losses))
        # torch.save(model.state_dict(), "./results/trainresult/model_{}itr_{}.chkpnt".format(it,losses))
        if (it % 50 == 0 and it != 0) or (it == args.num_iterations):
            model.eval()
            with torch.no_grad():
                print("validation:")
                tester = Tester(dataset, model, "valid", args.model)
                measure_valid, _ = tester.test()
                mrr = measure_valid.mrr["fil"]
                hit1 = measure_valid.hit1["fil"]
                measure_dict = vars(measure_valid)
                with open("./results/validresult/validresult.txt", "a") as f:
                    for k, v in measure_dict.items():
                        f.write("{}: {}\n".format(k, v))
                is_best_model = (best_model is None) or (mrr > best_model.best_mrr and hit1 > best_model.best_hit1)
                if is_best_model:
                    print("new hit1: {}".format(hit1))
                    print("new mrr: {}".format(mrr))
                    if best_model:
                        print("old hit1: {}".format(best_model.best_hit1))
                        print("old mrr: {}".format(best_model.best_mrr))
                    best_model = model
                    best_model.best_mrr.data = torch.from_numpy(np.array([mrr]))
                    best_model.best_itr.data = torch.from_numpy(np.array([it]))
                    best_model.best_hit1.data = torch.from_numpy(np.array([hit1]))
                    torch.save(best_model.state_dict(), "./results/trainresult/bestmodel.chkpnt")
                # Save the model at checkpoint
                # save_model(model=best_model, opt=opt, measure=measure_valid, measure_by_arity=None, args=args, test_by_arity=False, itr=it, test_or_valid="valid", is_best_model=is_best_model)

    with torch.no_grad():
        tester = Tester(dataset, best_model, "test", args.model)
        measure_arity, measure_by_arity = tester.test(test_by_arity=test_by_arity)
        with open("./results/testresult.txt","a") as ftest:
            measure_arity_dict = vars(measure_arity)
            for k, v in measure_arity_dict.items():
                ftest.write("{}: {}\n".format(k, v))




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, default="HyIE")
    parser.add_argument('-dataset', type=str, default="./data/FB-AUTO/")
    parser.add_argument('-lr', type=float, default=0.005)
    parser.add_argument('-nr', type=int, default=10)
    parser.add_argument('-filt_w', type=int, default=1)
    parser.add_argument('-filt_h', type=int, default=1)
    parser.add_argument('-emb_dim', type=int, default=400)
    parser.add_argument('-emb_dim1', type=int, default=20)
    parser.add_argument('-hidden_drop', type=float, default=0.2)
    parser.add_argument('-input_drop', type=float, default=0.2)
    parser.add_argument('-stride', type=int, default=2)
    parser.add_argument('-num_iterations', type=int, default=500)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-test_by_arity', type=bool, default=True)
    parser.add_argument("-test", action="store_true",
                        help="If -test is set, then you must specify a -pretrained model. "
                             + "This will perform testing on the pretrained model and save the output in -output_dir")
    parser.add_argument('-pretrained', type=str, default=None,
                        help="A path to a trained model (.chkpnt file), which will be loaded if provided.")
    parser.add_argument('-output_dir', type=str, default="./record/",
                        help="A path to the directory where the model will be saved and/or loaded from.")
    parser.add_argument('-restartable', action="store_true",
                        help="If restartable is set, you must specify an output_dir")
    args = parser.parse_args()

    main(args)
