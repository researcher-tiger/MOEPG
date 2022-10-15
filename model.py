from torch import nn
import torch.nn.functional as F
import numpy as np
import torch
from torch.nn import Module, Embedding, LSTM, Linear, Dropout

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)

device = "cuda" if torch.cuda.is_available() else "cpu"

class OnlineNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size):
        super(OnlineNet, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, n_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class TargetNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden_size):
        super(TargetNet, self).__init__()
        self.fc1 = nn.Linear(n_states, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(hidden_size, n_actions)
        self.out.weight.data.normal_(0, 0.1)
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

class MOEPG(object):
    def __init__(self, n_states, n_actions, memory_size, hidden_size, lr, batch_size):
        self.online_net, self.target_net = OnlineNet(n_states, n_actions, hidden_size), TargetNet(n_states, n_actions, hidden_size).to(device)

        self.learn_step_counter = 0

        self.memory_counter = 0

        self.memory = np.zeros((memory_size, n_states * 2 + 2))

        if torch.cuda.is_available():
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr, capturable=True)
        else:
            self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)

        self.loss_func = nn.MSELoss()

        self.epsilon = 0.99

        self.target_replace_iter = 100

        self.n_states = n_states

        self.n_actions = n_actions

        self.memory_size = memory_size

        self.hidden_size = hidden_size

        self.batch_size = batch_size

        self.gamma = 0.9

    def choose_action(self, x, partition_list, epsilon):
        x = torch.unsqueeze(x, 0, )

        if np.random.uniform() < self.epsilon:
            if partition_list != []:
                action = np.random.choice(list(set([i for i in range(self.n_actions)])-set(partition_list)))
            else:
                action = np.random.randint(0, self.n_actions)

        else:
            actions_value = self.online_net.forward(x)

            actions_value = actions_value.cpu().detach()

            if partition_list != []:
                actions_value[0][partition_list] = torch.min(actions_value)
                action = torch.argmax(actions_value).item()

            else:
                actions_value[0][torch.argmax(actions_value)] = torch.min(actions_value)
                action = torch.argmax(actions_value).item()

        if self.epsilon > epsilon:
            self.epsilon -= 0.01
        else:
            self.epsilon = epsilon

        return action

    def store_transition(self, s, a, r, s_):

        transition = np.hstack((s.cpu(), [a, r], s_.cpu()))

        index = self.memory_counter % self.memory_size

        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :self.n_states]).to(device)
        b_a = torch.LongTensor(b_memory[:, self.n_states:self.n_states+1].astype(int)).to(device)
        b_r = torch.FloatTensor(b_memory[:, self.n_states+1:self.n_states+2]).to(device)
        b_s_ = torch.FloatTensor(b_memory[:, -self.n_states:]).to(device)

        q_eval = self.online_net(b_s).gather(1, b_a)

        max_action = self.online_net(b_s_).max(1)[1]

        q_next = self.target_net(b_s_).gather(1, max_action.unsqueeze(1)).squeeze(1)

        q_target = b_r + self.gamma * q_next.view(self.batch_size, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

class DKT(Module):
    def __init__(self, num_q, emb_size, hidden_size):
        super(DKT, self).__init__()
        self.num_q = num_q
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.interaction_emb = Embedding(self.num_q * 2, self.emb_size)
        self.lstm_layer = LSTM(
            self.emb_size, self.hidden_size, batch_first=True
        )
        self.out_layer = Linear(self.hidden_size, self.num_q)

    def forward(self, q, r):
        qr = q + self.num_q * r
        h, _ = self.lstm_layer(self.interaction_emb(qr))
        y = self.out_layer(h)
        y = torch.sigmoid(y)
        return y

