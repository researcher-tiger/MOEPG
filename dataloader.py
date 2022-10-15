import os
import pickle
import numpy as np
import pandas as pd
import random
from torch.utils.data import Dataset
from utils import match_seq_len

class Statics2011(Dataset):
    def __init__(self, seq_len, u=None) -> None:
        super().__init__()
        self.u = u
        self.dataset_dir = "statics2011"
        self.seq_len = seq_len
        self.dataset_path = os.path.join(
            self.dataset_dir, "ds507_tx_All_Data_1664_2017_0227_034415.txt"
        )
        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "skill_cover.pkl"), "rb") as f:
                self.skill_cover = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "QB.pkl"), "rb") as f:
                self.QB = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "p2q.pkl"), "rb") as f:
                self.p2q = pickle.load(f)

            if self.u != None:
                self.q_seqs, self.r_seqs = self.preprocess_u()
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, \
            self.u2idx, self.QB, self.skill_cover, self.p2q = self.preprocess()

        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]

        if self.seq_len:
            self.q_seqs, self.r_seqs = \
                match_seq_len(self.q_seqs, self.r_seqs, self.seq_len)

        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        print("preprocess statics2011")
        df = pd.read_csv(self.dataset_path, sep="\t", low_memory=False) \
            .dropna(subset=["Problem Name", "Step Name", "Outcome"]) \
            .sort_values(by=["Time"])
        df = df[df["Attempt At Step"] == 1]
        df = df[df["Student Response Type"] == "ATTEMPT"]
        kcs = []
        for _, row in df.iterrows():
            kcs.append("{}_{}".format(row["Problem Name"], row["Step Name"]))
        df["KC"] = kcs
        u_list = np.unique(df["Anon Student Id"].values)
        q_list = np.unique(df["KC"].values)
        p_list = np.unique(df["Problem Name"].values)
        QB = list(p_list)
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}

        p2q = {p: [] for p in p_list}
        for p in p_list:
            q = set(list(df[df["Problem Name"] == p]["KC"]))
            for q_ in q:
                p2q[p].append(q2idx[q_])
        for i in range(1700):
            temp_p = random.choice(QB)
            QB.append(f"test_{i}")
            p2q[f"test_{i}"] = p2q[temp_p]
        skill_cover_dic = {q2idx[q]: 0 for q in q_list}
        sum_q = 0
        for p in QB:
            for q in p2q[p]:
                skill_cover_dic[q] += 1
                sum_q += 1
        skill_cover = []
        for skill in skill_cover_dic:
            skill_cover.append(skill_cover_dic[skill] / sum_q)
        q_seqs = []
        r_seqs = []
        for u in u_list:
            u_df = df[df["Anon Student Id"] == u]
            q_seqs.append([q2idx[q] for q in u_df["KC"].values])
            r_seqs.append((u_df["Outcome"].values == "CORRECT").astype(int))
        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        with open(os.path.join(self.dataset_dir, "QB.pkl"), "wb") as f:
            pickle.dump(QB, f)
        with open(os.path.join(self.dataset_dir, "skill_cover.pkl"), "wb") as f:
            pickle.dump(skill_cover, f)
        with open(os.path.join(self.dataset_dir, "p2q.pkl"), "wb") as f:
            pickle.dump(p2q, f)
        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx, QB, skill_cover, p2q

    def preprocess_u(self):
        df = pd.read_csv(self.dataset_path, sep="\t", low_memory=False) \
            .dropna(subset=["Problem Name", "Step Name", "Outcome"]) \
            .sort_values(by=["Time"])
        df = df[df["Attempt At Step"] == 1]
        df = df[df["Student Response Type"] == "ATTEMPT"]
        kcs = []
        for _, row in df.iterrows():
            kcs.append("{}_{}".format(row["Problem Name"], row["Step Name"]))
        df["KC"] = kcs
        q_seqs = []
        r_seqs = []
        u_df = df[df["Anon Student Id"] == self.u]
        q_seqs.append([self.q2idx[q] for q in u_df["KC"].values])
        r_seqs.append((u_df["Outcome"].values == "CORRECT").astype(int))

        return q_seqs, r_seqs

class ASSIST2009(Dataset):
    def __init__(self, seq_len, u=None) -> None:
        super().__init__()
        self.dataset_dir = "assistments0910"
        self.dataset_path = os.path.join(
            self.dataset_dir, "skill_builder_data.csv"
        )
        self.u = u

        if os.path.exists(os.path.join(self.dataset_dir, "q_seqs.pkl")):
            with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "rb") as f:
                self.q_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "rb") as f:
                self.r_seqs = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q_list.pkl"), "rb") as f:
                self.q_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u_list.pkl"), "rb") as f:
                self.u_list = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "rb") as f:
                self.q2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "rb") as f:
                self.u2idx = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "skill_cover.pkl"), "rb") as f:
                self.skill_cover = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "QB.pkl"), "rb") as f:
                self.QB = pickle.load(f)
            with open(os.path.join(self.dataset_dir, "p2q.pkl"), "rb") as f:
                self.p2q = pickle.load(f)

            if self.u != None:
                self.q_seqs, self.r_seqs = self.preprocess_u()
        else:
            self.q_seqs, self.r_seqs, self.q_list, self.u_list, self.q2idx, self.u2idx, \
            self.QB, self.skill_cover, self.p2q = self.preprocess()
        self.num_u = self.u_list.shape[0]
        self.num_q = self.q_list.shape[0]
        if seq_len:
            self.q_seqs, self.r_seqs = match_seq_len(self.q_seqs, self.r_seqs, seq_len)
        self.len = len(self.q_seqs)

    def __getitem__(self, index):
        return self.q_seqs[index], self.r_seqs[index]

    def __len__(self):
        return self.len

    def preprocess(self):
        df = pd.read_csv(self.dataset_path, encoding='unicode_escape', low_memory=False).dropna(subset=["skill_name"]).drop_duplicates(subset=["order_id", "skill_name"]).sort_values(by=["order_id"])
        u_list = np.unique(df["user_id"].values)
        for u in u_list:
            if len(df[df["user_id"] == u]) < 3:
                df = df.drop(df[df["user_id"] == u].index)
        u_list = np.unique(df["user_id"].values)
        q_list = np.unique(df["skill_name"].values)
        p_list = np.unique(df["problem_id"].values)
        p_list = list(p_list)
        u2idx = {u: idx for idx, u in enumerate(u_list)}
        q2idx = {q: idx for idx, q in enumerate(q_list)}
        p2q = {p: [] for p in p_list}
        for p in p_list:
            q = set(list(df[df["problem_id"] == p]["skill_name"]))
            for q_ in q:
                p2q[p].append(q2idx[q_])
        q_seqs = []
        r_seqs = []
        for u in u_list:
            df_u = df[df["user_id"] == u]
            q_seq = [q2idx[q] for q in df_u["skill_name"]]
            r_seq = df_u["correct"].values
            q_seqs.append(q_seq)
            r_seqs.append(r_seq)
        QB = random.sample(p_list, 10000)
        skill_cover_dic = {q2idx[q]: 0 for q in q_list}
        sum_q = 0
        for p in QB:
            for q in p2q[p]:
                skill_cover_dic[q] += 1
                sum_q += 1
        skill_cover = []
        for skill in skill_cover_dic:
            skill_cover.append(skill_cover_dic[skill] / sum_q)
        with open(os.path.join(self.dataset_dir, "q_seqs.pkl"), "wb") as f:
            pickle.dump(q_seqs, f)
        with open(os.path.join(self.dataset_dir, "r_seqs.pkl"), "wb") as f:
            pickle.dump(r_seqs, f)
        with open(os.path.join(self.dataset_dir, "q_list.pkl"), "wb") as f:
            pickle.dump(q_list, f)
        with open(os.path.join(self.dataset_dir, "u_list.pkl"), "wb") as f:
            pickle.dump(u_list, f)
        with open(os.path.join(self.dataset_dir, "q2idx.pkl"), "wb") as f:
            pickle.dump(q2idx, f)
        with open(os.path.join(self.dataset_dir, "u2idx.pkl"), "wb") as f:
            pickle.dump(u2idx, f)
        with open(os.path.join(self.dataset_dir, "QB.pkl"), "wb") as f:
            pickle.dump(QB, f)
        with open(os.path.join(self.dataset_dir, "skill_cover.pkl"), "wb") as f:
            pickle.dump(skill_cover, f)
        with open(os.path.join(self.dataset_dir, "p2q.pkl"), "wb") as f:
            pickle.dump(p2q, f)
        return q_seqs, r_seqs, q_list, u_list, q2idx, u2idx, QB, skill_cover, p2q

    def preprocess_u(self):
        df = pd.read_csv(self.dataset_path, encoding='unicode_escape', low_memory=False).dropna(
            subset=["skill_name"]).drop_duplicates(subset=["order_id", "skill_name"]).sort_values(by=["order_id"])
        q_seqs = []
        r_seqs = []
        df_u = df[df["user_id"] == self.u]
        q_seq = [self.q2idx[q] for q in df_u["skill_name"]]
        r_seq = df_u["correct"].values
        q_seqs.append(q_seq)
        r_seqs.append(r_seq)
        return q_seqs, r_seqs
