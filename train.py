import argparse
import copy
import pickle
import random
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans
from torch.utils.data import random_split, DataLoader
from torch.nn.functional import one_hot, binary_cross_entropy
import os
import torch
from tqdm import tqdm
from model import MOEPG
from dataloader import Statics2011
from dataloader import ASSIST2009
from model import DKT
from utils import collate_fn
from torch.optim import Adam
from scipy import stats
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser()

parser.add_argument('--random_seed', type=int, default=22)

# MOEPG parameter
parser.add_argument('--rl_hidden_size', type=int, default=200)
parser.add_argument('--memory_size', type=int, default=2000)
parser.add_argument('--rl_batch_size', type=int, default=128)
parser.add_argument('--rl_max_episode', type=int, default=5000)
parser.add_argument('--rl_learning_rate', type=float, default=0.01)
parser.add_argument('--q_embedding_size', type=int, default=30)
parser.add_argument('--epsilon', type=float, default=0.1)
parser.add_argument('--dataset', type=str, default="statics2011", help="assistments0910|statics2011")

device = "cuda" if torch.cuda.is_available() else "cpu"

def del_tensor_ele(arr,index):
    arr1 = arr[0:index]
    arr2 = arr[index+1:]
    return torch.cat((arr1, arr2), dim=0)

def get_students():
    if os.path.exists(os.path.join(path, "dk_list.pkl")):
        with open(os.path.join(path, "dk_list.pkl"), "rb") as f:
            dk_list = pickle.load(f)
    else:
        with open(os.path.join(path, "u_list.pkl"), "rb") as f:
            u_list = pickle.load(f)
        u_list = list(u_list)
        select_u_list = random.sample(u_list, 50)
        dk_list = []
        for u in tqdm(select_u_list):
            if path == "assistments0910":
                dataset = ASSIST2009(100, u)
            else:
                dataset = Statics2011(100, u)
            test_loader = DataLoader(
                dataset, batch_size=len(dataset), shuffle=False,
                collate_fn=collate_fn)

            with torch.no_grad():
                for data in test_loader:
                    q, r, t, d, m = data

                    kt_model.eval()

                    y = kt_model(q, r)

                    for i in range(len(m[-1].tolist())):
                        if m[-1].tolist()[i]:
                            dk = y[-1][i]
                    dk_list.append(dk.tolist())

        with open(os.path.join(path, "dk_list.pkl"), "wb") as f:
            pickle.dump(dk_list, f)

    return dk_list

def get_paper_score(paper):
    score_list = []
    for user_dk in dk_list:
        score = 0
        for p in paper:
            p = idx2p[p]
            temp_score = 1
            for q in p2q[p]:
                temp_score *= user_dk[q]
            score += temp_score
        score_list.append(score)
    return score_list

def get_reward(paper):
    skill_cover_paper = []
    skill_cover_paper_dic = {q: 0 for q in range(num_q)}
    sum_q = 0
    for p in paper.detach().numpy():
        for q in p2q[idx2p[p]]:
            skill_cover_paper_dic[q] += 1
            sum_q += 1
    for skill in skill_cover_paper_dic:
        skill_cover_paper.append(skill_cover_paper_dic[skill] / sum_q)
    r3 = cosine_similarity([skill_cover], [skill_cover_paper])[0][0]
    score_list = get_paper_score(paper.detach().numpy().tolist())
    r1 = 1 - abs(np.mean(score_list) - 70) / len(paper.detach().numpy().tolist())
    r2 = 1 - stats.wasserstein_distance(paper_distribution, score_list) / len(paper)
    r = r1/3 + r2/3 + r3/3
    return r, r3

def get_state(paper):
    paper = torch.sort(paper).values
    paper_state = []
    for p in paper:
        paper_state.append(torch.mean(kt_model.interaction_emb(torch.tensor(p2q[idx2p[p.item()]])), dim=0).cpu().detach().numpy().tolist())
    return torch.tensor(paper_state).reshape(-1)

if __name__ == "__main__":
    args = parser.parse_args()
    seed = args.random_seed
    random.seed(seed)
    path = args.dataset
    dataset = None
    train_loader = None
    test_loader = None
    if args.dataset == "assistments0910":
        kt_train_ratio = 0.8
        dataset = ASSIST2009(100)
        train_size = int(len(dataset) * kt_train_ratio)
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator(device=device)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=test_size, shuffle=False,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
    elif args.dataset == "statics2011":
        kt_train_ratio = 0.7

        dataset = Statics2011(100)
        train_size = int(len(dataset) * kt_train_ratio)
        test_size = len(dataset) - train_size

        train_dataset, test_dataset = random_split(
            dataset, [train_size, test_size], generator=torch.Generator(device=device)
        )

        train_loader = DataLoader(
            train_dataset, batch_size=64, shuffle=False,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
        test_loader = DataLoader(
            test_dataset, batch_size=test_size, shuffle=False,
            collate_fn=collate_fn, generator=torch.Generator(device=device)
        )
    else:
        print("input error")
        exit(0)

    # kt model
    kt_model = DKT(dataset.num_q, args.q_embedding_size, 100).to(device)

    if not os.path.exists(os.path.join(path, "dkt_model.ckpt")):
        print("train kt_model")
        # train kt_model
        if torch.cuda.is_available():
            opt = Adam(kt_model.parameters(), 0.001, capturable=True)
        else:
            opt = Adam(kt_model.parameters(), 0.001)
        max_auc = 0
        for episode in range(100):
            for data in train_loader:
                q, r, t, d, m = data
                kt_model.train()
                y = kt_model(q, r)

                y = (y * one_hot(d, kt_model.num_q)).sum(-1)

                y = torch.masked_select(y, m)

                t = torch.masked_select(t, m)

                opt.zero_grad()

                loss = binary_cross_entropy(y, t)

                loss.backward()

                opt.step()

            with torch.no_grad():
                for data in test_loader:
                    q, r, t, d, m = data

                    kt_model.eval()

                    y = kt_model(q, r)

                    y = (y * one_hot(d, kt_model.num_q)).sum(-1)

                    y = torch.masked_select(y, m).detach().cpu()
                    t = torch.masked_select(t, m).detach().cpu()
                    auc = metrics.roc_auc_score(
                        y_true=t.numpy(), y_score=y.numpy()
                    )

                    if auc > max_auc:
                        max_auc = auc
                        torch.save(kt_model.state_dict(), os.path.join(path, "dkt_model.ckpt"))

                    print("Episode: {},   AUC: {},   MAX_AUC: {}".format(episode, auc, max_auc))

    else:
        kt_model.load_state_dict(torch.load(os.path.join(path, "dkt_model.ckpt")))

    QB = dataset.QB

    p2idx = {p: idx for idx, p in enumerate(QB)}

    idx2p = {idx: p for idx, p in enumerate(QB)}

    p2q = dataset.p2q

    if not os.path.exists(os.path.join(path, "partition_dic.pkl")):
        # partition
        QB_embedd = []
        for p in QB:
            QB_embedd.append(torch.mean(kt_model.interaction_emb(torch.tensor(p2q[p])), dim=0).cpu().detach().numpy().tolist())

        kmeans = KMeans(n_clusters=10, random_state=seed).fit(QB_embedd)

        labels = kmeans.labels_

        labels = labels.tolist()

        partition_dic = {i: [] for i in range(10)}

        for i in range(len(QB)):
            partition_dic[labels[i]].append(p2idx[QB[i]])

        with open(os.path.join(path, "partition_dic.pkl"), "wb") as f:
            pickle.dump(partition_dic, f)
    else:
        with open(os.path.join(path, "partition_dic.pkl"), "rb") as f:
            partition_dic = pickle.load(f)

    num_q = dataset.num_q

    skill_cover = dataset.skill_cover

    n_actions = len(QB)

    n_states = args.q_embedding_size * 100

    dk_list = get_students()

    # MOEPG
    # n_states, n_actions, memory_size, hidden_size, lr, batch_size
    moepg = MOEPG(n_states, n_actions, args.memory_size, args.rl_hidden_size, args.rl_learning_rate, args.rl_batch_size)

    # The ideal distribution of student score
    X = stats.truncnorm((0 - 70) / 15, (100 - 70) / 15, loc=70, scale=15)
    paper_distribution = X.rvs(100, random_state=seed)

    for paperindex in range(1, 21):
        print(f"paper {paperindex}")

        # random initialize
        paper = torch.LongTensor([p2idx[p] for p in random.sample(QB, 100)])
        bestpaper = paper
        r_, _ = get_reward(paper)
        max_reward = r_

        for episode in range(args.rl_max_episode):
            paper_state = get_state(paper).to(device)
            paper_ = copy.deepcopy(paper)

            y_suit = []
            for index in range(100):
                temp_paper = copy.deepcopy(paper_)
                temp_paper = del_tensor_ele(temp_paper, index)
                r, _ = get_reward(temp_paper)
                y_suit.append(r)

            replace_index = np.argmax(y_suit)

            partition_list = []
            for index in partition_dic:
                if paper[replace_index] in partition_dic[index]:
                    partition_list = partition_dic[index]
                    break

            a = moepg.choose_action(paper_state, partition_list, args.epsilon)

            if a not in paper_:
                paper_[replace_index] = a

            r, _ = get_reward(paper_)

            if r > max_reward:
                max_reward = r
                bestpaper = paper_

            if r <= r_:
                r_ = r
                r += -10
            else:
                r_ = r

            paper = paper_

            paper_state_ = get_state(paper)

            if (episode+1) % 200 == 0:
                print(f"episode: {episode+1}, reward: {r_}")

            moepg.store_transition(paper_state, a, r, paper_state_)

            if moepg.memory_counter > moepg.batch_size:
                moepg.learn()

        print(f"The optimal test paper at the current episode is {bestpaper}, The maximum reward is {max_reward}")
        # save
        with open(os.path.join(path, f"best_paper_{paperindex}.pkl"), "wb") as f:
            pickle.dump(bestpaper, f)