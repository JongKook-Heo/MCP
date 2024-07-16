import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
device = 'cuda'

def gen_net(in_size=1, out_size=1, hidden_size=128, n_layers=3, activation='tanh'):
    net = []
    for i in range(n_layers):
        net.append(nn.Linear(in_size, hidden_size))
        net.append(nn.LeakyReLU())
        in_size = hidden_size
    net.append(nn.Linear(in_size, out_size))
    if activation == 'tanh':
        net.append(nn.Tanh())
    elif activation =='sig':
        net.append(nn.Sigmoid())
    else:
        net.append(nn.ReLU())
    
    return net

class RewardModel:
    def __init__(self, obs_dim, action_dim,
                 ensemble_size=3, lr=3e-4, mb_size=128, size_segment=1, max_size=100,
                 activation='tanh', capacity=5e5, large_batch=1, label_margin=0.0,
                 teacher_beta=-1, teacher_gamma=1, teacher_eps_mistake=0,
                 teacher_eps_skip=0, teacher_eps_equal=0):
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.lr = lr
        self.ensemble = []
        self.paramlst = []
        self.optimizer = None
        self.max_size = max_size
        self.activation = activation
        self.size_segment = size_segment
        
        self.capacity = int(capacity)
        self.buffer_seg1 = np.empty((self.capacity, size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_seg2 = np.empty((self.capacity, size_segment, self.obs_dim + self.action_dim), dtype=np.float32)
        self.buffer_label = np.empty((self.capacity, 1), dtype=np.float32)
        self.buffer_index = 0
        self.buffer_full = False
        
        self.construct_ensemble()
        self.inputs = []
        self.targets = []
        self.mb_size = mb_size
        self.origin_mb_size = mb_size
        self.train_batch_size = 128
        self.CELoss = nn.CrossEntropyLoss()
        self.large_batch = large_batch
        
        # new teacher
        self.teacher_beta = teacher_beta
        self.teacher_gamma = teacher_gamma
        self.teacher_eps_mistake = teacher_eps_mistake
        self.teacher_eps_equal = teacher_eps_equal
        self.teacher_eps_skip = teacher_eps_skip
        self.teacher_thres_skip = 0
        self.teacher_thres_equal = 0
        
        self.label_margin = label_margin
        self.label_target = 1-2*self.label_margin
        
    def eval(self):
        for model in self.ensemble:
            model.eval()
    
    def construct_ensemble(self):
        for i in range(self.ensemble_size):
            model = nn.Sequential(*gen_net(in_size=self.obs_dim + self.action_dim,
                                           out_size=1, hidden_size=256, n_layers=3,
                                           activation=self.activation)).float().to(device)
            self.ensemble.append(model)
            self.paramlst.extend(model.parameters())
        self.optimizer = torch.optim.Adam(self.paramlst, lr=self.lr)
        
    def softXEnt_loss(self, input, target):
        """_summary_

        Args:
            input (_type_): _description_
            target (_type_): _description_
        """
        log_probs = torch.nn.functional.log_softmax(input, dim=1)
        return - (target * log_probs).sum() / input.shape[0]
    
    def change_batch(self, new_frac):
        self.mb_size = int(self.origin_mb_size*new_frac)
    
    def set_batch(self, new_batch):
        self.mb_size = int(new_batch)
        
    def set_teacher_thres_skip(self, new_margin):
        self.teacher_thres_skip = new_margin * self.teacher_eps_skip
    
    def set_teacher_thres_equal(self, new_margin):
        self.teacher_thres_equal = new_margin * self.teacher_eps_equal
    
    def add_data(self, obs, action, reward, done):
        sa_t = np.concatenate([obs, action], axis=-1)
        r_t = reward
        
        flat_input = sa_t.reshape(1, self.obs_dim + self.action_dim)
        flat_target = np.array(r_t).reshape(1, 1)
        
        init_data = len(self.inputs) == 0
        if init_data:
            self.inputs.append(flat_input)
            self.targets.append(flat_target)
            
        elif done:
            self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
            self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
            
            if len(self.inputs) > self.max_size:
                self.inputs = self.inputs[1:]
                self.targets = self.targets[1:]
            self.inputs.append([]) # new episode
            self.targets.append([])
        
        else:
            if len(self.inputs[-1]) == 0: #new episode
                self.inputs[-1] = flat_input
                self.targets[-1] = flat_target
            else:
                self.inputs[-1] = np.concatenate([self.inputs[-1], flat_input])
                self.targets[-1] = np.concatenate([self.targets[-1], flat_target])
    
    def get_rank_probability(self, x_1, x_2):
        """
        x_1, x_2 : (mb_size * large_batch, segment_length, obs_dim + action_dim)
        """
        # get probability x_1 > x_2
        probs = []
        for member in range(self.ensemble_size):
            probs.append(self.p_hat_member(x_1, x_2, member=member).cpu().numpy())
        probs = np.array(probs)
        return np.mean(probs, axis=0), np.std(probs, axis=0)
    
    def get_entropy(self, x_1, x_2):
        ent = []
        for member in range(self.ensemble_size):
            ent.append(self.p_hat_entropy(x_1, x_2, member=member).cpu().numpy())
        ent = np.array(ent)
        return np.mean(ent, axis=0), np.std(ent, axis=0)
    
    def p_hat_member(self, x_1, x_2, member=-1):
        """
        x_1, x_2 : (mb_size * large_batch, segment_length, obs_dim + action_dim)
        return (mb_size * large_batch, 1)
        """
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member) # (mb_size * large_batch, segment_length, 1)
            r_hat2 = self.r_hat_member(x_2, member=member) # (mb_size * large_batch, segment_length, 1)
            r_hat1 = r_hat1.sum(axis=1) # (mb_size * larget_batch, 1)
            r_hat2 = r_hat2.sum(axis=1) # (mb_size * larget_batch, 1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        return F.softmax(r_hat, dim=-1)[:, 0]
    
    def p_hat_entropy(self, x_1, x_2, member=-1):
        """
        x_1, x_2 : (mb_size * large_batch, segment_length, obs_dim + action_dim)
        """
        with torch.no_grad():
            r_hat1 = self.r_hat_member(x_1, member=member) # (mb_size * large_batch, segment_length, 1)
            r_hat2 = self.r_hat_member(x_2, member=member) # (mb_size * large_batch, segment_length, 1)
            r_hat1 = r_hat1.sum(axis=1) # (mb_size * larget_batch, 1)
            r_hat2 = r_hat2.sum(axis=1) # (mb_size * larget_batch, 1)
            r_hat = torch.cat([r_hat1, r_hat2], axis=-1)
        ent = F.softmax(r_hat, dim=-1) * F.log_softmax(r_hat, dim=-1)
        ent = ent.sum(axis=-1).abs()
        return ent
    
    def r_hat_member(self, x, member=-1):
        if isinstance(x, np.ndarray):
            return self.ensemble[member](torch.from_numpy(x).float().to(device))
        else:
            return self.ensemble[member](x.float().to(device))
    
    def r_hat(self, x):
        """
        1) x : (obs_dim + action_dim, ) -> (ensemble_size, 1) -> (1, )
        """
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats)
    
    def r_hat_batch(self, x):
        """
        x : (batch_size, obs_dim + action_dim) -> (ensemble_size, batch_size, 1) -> (batch_size, 1)
        """
        r_hats = []
        for member in range(self.ensemble_size):
            r_hats.append(self.r_hat_member(x, member=member).detach().cpu().numpy())
        r_hats = np.array(r_hats)
        return np.mean(r_hats, axis=0)
    
    def save(self, model_dir, step):
        for member in range(self.ensemble_size):
            torch.save(self.ensemble[member].state_dict(), '%s/reward_model_%s_%s.pt'%(model_dir, step, member))
    
    def load(self, model_dir, step):
        for member in range(self.ensemble_size):
            self.ensemble[member].load_state_dict(torch.load('%s/reward_model_%s_%s.pt'%(model_dir, step, member)))
    
    # def get_train_acc(self):
    #     pass
    
    def get_queries(self, mb_size=20):
        len_traj, max_len = len(self.inputs[0]), len(self.inputs)
        
        if len(self.inputs[-1]) < len_traj:
            max_len = max_len - 1
        
        train_inputs = np.array(self.inputs[:max_len])
        train_targets = np.array(self.targets[:max_len])
        
        batch_index_1 = np.random.choice(max_len, size=mb_size, replace=True)
        batch_index_2 = np.random.choice(max_len, size=mb_size, replace=True)
        
        sa_t_1 = train_inputs[batch_index_1] # (batch_size, episode_len(T), obs_dim + action_dim)
        r_t_1 = train_targets[batch_index_1] # (batch_size, episode_len(T), 1)
        sa_t_2 = train_inputs[batch_index_2] # (batch_size, episode_len(T), obs_dim + action_dim)
        r_t_2 = train_targets[batch_index_2] # (batch_size, episode_len(T), 1)
        
        sa_t_1 = sa_t_1.reshape(-1, sa_t_1.shape[-1]) # (batch_size * T, obs_dim + action_dim)
        r_t_1 = r_t_1.reshape(-1, r_t_1.shape[-1]) # (batch_size * T, 1)
        sa_t_2 = sa_t_2.reshape(-1, sa_t_2.shape[-1]) # (batch_size * T, obs_dim + action_dim)
        r_t_2 = r_t_2.reshape(-1, r_t_2.shape[-1]) # (batch_size * T, 1)
        
        time_index = np.array([list(range(i*len_traj, i*len_traj+self.size_segment)) for i in range(mb_size)])
        # mb_size 128, len_traj 500, segment_len 50
        # time_index : [[0~49], [500~549], ..., [63500~63549]]
        
        time_index_1 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        time_index_2 = time_index + np.random.choice(len_traj - self.size_segment, size=mb_size, replace=True).reshape(-1, 1)
        # time_index 1 &2 -> (batch_size, size_segment)
        
        sa_t_1 = np.take(sa_t_1, time_index_1, axis=0) # Batch x size_seg x dim of s&a
        r_t_1 = np.take(r_t_1, time_index_1, axis=0) # Batch x size_seg x 1
        sa_t_2 = np.take(sa_t_2, time_index_2, axis=0) # Batch x size_seg x dim of s&a
        r_t_2 = np.take(r_t_2, time_index_2, axis=0) # Batch x size_seg x 1
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2
    
    def put_queries(self, sa_t_1, sa_t_2, labels):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        total_sample = sa_t_1.shape[0]
        next_index = self.buffer_index + total_sample
        if next_index >= self.capacity:
            self.buffer_full = True
            maximum_index = self.capacity - self.buffer_index
            np.copyto(self.buffer_seg1[self.buffer_index:self.capacity], sa_t_1[:maximum_index])
            np.copyto(self.buffer_seg2[self.buffer_index:self.capacity], sa_t_2[:maximum_index])
            np.copyto(self.buffer_label[self.buffer_index:self.capacity], labels[:maximum_index])

            remain = total_sample - (maximum_index)
            if remain > 0:
                np.copyto(self.buffer_seg1[0:remain], sa_t_1[maximum_index:])
                np.copyto(self.buffer_seg2[0:remain], sa_t_2[maximum_index:])
                np.copyto(self.buffer_label[0:remain], labels[maximum_index:])

            self.buffer_index = remain
        else:
            np.copyto(self.buffer_seg1[self.buffer_index:next_index], sa_t_1)
            np.copyto(self.buffer_seg2[self.buffer_index:next_index], sa_t_2)
            np.copyto(self.buffer_label[self.buffer_index:next_index], labels)
            self.buffer_index = next_index
            
    def get_label(self, sa_t_1, sa_t_2, r_t_1, r_t_2):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        return label : (batch_size, 1)
        """
        sum_r_t_1 = np.sum(r_t_1, axis=1)
        sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # skip the query
        if self.teacher_thres_skip > 0: 
            max_r_t = np.maximum(sum_r_t_1, sum_r_t_2)
            max_index = (max_r_t > self.teacher_thres_skip).reshape(-1)
            if sum(max_index) == 0:
                return None, None, None, None, []

            sa_t_1 = sa_t_1[max_index]
            sa_t_2 = sa_t_2[max_index]
            r_t_1 = r_t_1[max_index]
            r_t_2 = r_t_2[max_index]
            sum_r_t_1 = np.sum(r_t_1, axis=1)
            sum_r_t_2 = np.sum(r_t_2, axis=1)
        
        # equally preferable
        margin_index = (np.abs(sum_r_t_1 - sum_r_t_2) < self.teacher_thres_equal).reshape(-1)
        
        # perfectly rational
        seg_size = r_t_1.shape[1]
        temp_r_t_1 = r_t_1.copy()
        temp_r_t_2 = r_t_2.copy()
        for index in range(seg_size-1):
            temp_r_t_1[:,:index+1] *= self.teacher_gamma
            temp_r_t_2[:,:index+1] *= self.teacher_gamma
        sum_r_t_1 = np.sum(temp_r_t_1, axis=1)
        sum_r_t_2 = np.sum(temp_r_t_2, axis=1)
            
        rational_labels = 1*(sum_r_t_1 < sum_r_t_2)
        if self.teacher_beta > 0: # Bradley-Terry rational model
            r_hat = torch.cat([torch.Tensor(sum_r_t_1), 
                               torch.Tensor(sum_r_t_2)], axis=-1)
            r_hat = r_hat*self.teacher_beta
            ent = F.softmax(r_hat, dim=-1)[:, 1]
            labels = torch.bernoulli(ent).int().numpy().reshape(-1, 1)
        else:
            labels = rational_labels
        
        # making a mistake
        len_labels = labels.shape[0]
        rand_num = np.random.rand(len_labels)
        noise_index = rand_num <= self.teacher_eps_mistake
        labels[noise_index] = 1 - labels[noise_index]
 
        # equally preferable
        labels[margin_index] = -1 
        
        return sa_t_1, sa_t_2, r_t_1, r_t_2, labels
        
    def uniform_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size = self.mb_size)
        sa_t_2, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def disagreement_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 = self.get_queries(mb_size = self.mb_size * self.large_batch)
        _, disagree = self.get_rank_probability(sa_t_1, sa_t_2)
        top_k_index = (-disagree).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels) 
        return len(labels)
    
    def entropy_sampling(self):
        """
        sa_t : (batch_size, size_segment, obs_dim + action_dim)
        r_t : (batch_size, size_segment, 1)
        label : (batch_size, 1), if equally preferable -> -1??
        """
        sa_t_1, sa_t_2, r_t_1, r_t_2 =  self.get_queries(mb_size=self.mb_size*self.large_batch)
        entropy, _ = self.get_entropy(sa_t_1, sa_t_2)
        top_k_index = (-entropy).argsort()[:self.mb_size]
        r_t_1, sa_t_1 = r_t_1[top_k_index], sa_t_1[top_k_index]
        r_t_2, sa_t_2 = r_t_2[top_k_index], sa_t_2[top_k_index]
        sa_t_1, sa_t_2, r_t_1, r_t_2, labels = self.get_label(sa_t_1, sa_t_2, r_t_1, r_t_2)
        if len(labels) > 0:
            self.put_queries(sa_t_1, sa_t_2, labels)
        return len(labels)
    
    def train_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1) # (batch_size, 1)
                r_hat2 = r_hat2.sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) # (batch_size, 2)
                
                curr_loss = self.CELoss(r_hat, labels)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
                
            loss.backward()
            self.optimizer.step()
            
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
    
    def train_soft_reward(self):
        ensemble_losses = [[] for _ in range(self.ensemble_size)]
        ensemble_acc = np.array([0 for _ in range(self.ensemble_size)])
        max_len = self.capacity if self.buffer_full else self.buffer_index
        total_batch_index = []
        for _ in range(self.ensemble_size):
            total_batch_index.append(np.random.permutation(max_len))
        
        num_iters = int(np.ceil(max_len/self.train_batch_size))
        total = 0
        for it in range(num_iters):
            self.optimizer.zero_grad()
            loss = 0.0
            
            last_index = (it + 1) * self.train_batch_size
            if last_index > max_len:
                last_index = max_len
            
            for member in range(self.ensemble_size):
                
                idxs = total_batch_index[member][it * self.train_batch_size:last_index]
                sa_t_1 = self.buffer_seg1[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                sa_t_2 = self.buffer_seg2[idxs] # (batch_size, size_segment, obs_dim + action_dim)
                labels = self.buffer_label[idxs] # (batch_size, 1)
                labels = torch.from_numpy(labels.flatten()).long().to(device)
                
                if member == 0:
                    total += labels.size(0)
                    
                r_hat1 = self.r_hat_member(sa_t_1, member=member) # (batch_size, size_segment, 1)
                r_hat2 = self.r_hat_member(sa_t_2, member=member) # (batch_size, size_segment, 1)
                r_hat1 = r_hat1.sum(axis=1) # (batch_size, 1)
                r_hat2 = r_hat2.sum(axis=1) # (batch_size, 1)
                r_hat = torch.cat([r_hat1, r_hat2], axis=-1) # (batch_size, 2)
                
                uniform_index = labels == -1
                labels[uniform_index] = 0
                target_onehot = torch.zeros_like(r_hat).scatter(1, labels.unsqueeze(1), self.label_target)
                target_onehot += self.label_margin
                if sum(uniform_index) > 0:
                    target_onehot[uniform_index] = 0.5
                curr_loss = self.softXEnt_loss(r_hat, target_onehot)
                loss += curr_loss
                ensemble_losses[member].append(curr_loss.item())
                
                _, predicted = torch.max(r_hat.data, 1) # (batch_size, )
                correct = (predicted == labels).sum().item()
                ensemble_acc[member] += correct
            
            loss.backward()
            self.optimizer.step()
        
        ensemble_acc = ensemble_acc / total
        return ensemble_acc
