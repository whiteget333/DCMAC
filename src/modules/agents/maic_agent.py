import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.distributions as D
from torch.distributions import kl_divergence, Categorical

def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)

class MAICAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(MAICAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.latent_dim = args.latent_dim
        self.n_actions = args.n_actions
        self.input_shape = input_shape
        self.tiny_msg_size = args.tiny_msg_size

        NN_HIDDEN_SIZE = args.nn_hidden_size
        activation_func = nn.LeakyReLU()

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

        self.encode_dim = args.encode_dim
        self.input_qkv = nn.Linear(input_shape, args.rnn_hidden_dim * 3)
        # self.input_qkv = nn.Linear(input_shape, args.rnn_hidden_dim * 5)
        '''
        self.q_proj = nn.Linear(input_shape, args.rnn_hidden_dim * 2)
        self.k_proj = nn.Linear(input_shape, args.rnn_hidden_dim * 2)
        self.v_proj = nn.Linear(input_shape, args.rnn_hidden_dim )
        
        self.lambda_init = lambda_init_fn(1)
        self.lambda_q1 = nn.Parameter(torch.zeros(args.rnn_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(args.rnn_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(args.rnn_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(args.rnn_hidden_dim, dtype=torch.float32).normal_(mean=0,std=0.1))
        '''
        # self.subln = RMSNorm(2 * args.rnn_hidden_dim, eps=1e-5, elementwise_affine=True)

        self.tiny_msg_net = nn.Linear(args.rnn_hidden_dim, args.tiny_msg_size)
        # self.tiny_msg_net = self.msg_net = nn.Sequential(
        #     nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
        #     activation_func,
        #     nn.Linear(NN_HIDDEN_SIZE, args.tiny_msg_size)
        # )

        self.demand_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2)
        )

        self.inference_net = nn.Sequential(
            nn.Linear(args.tiny_msg_size, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_agents * args.latent_dim * 2)
        )

        self.demand_inference_net = nn.Sequential(
            nn.Linear(args.rnn_hidden_dim + args.n_actions, NN_HIDDEN_SIZE),
            nn.BatchNorm1d(NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.latent_dim * 2)
        )

        # self.decoder_net = nn.Sequential(
        #     nn.Linear(args.tiny_msg_size, NN_HIDDEN_SIZE),
        #     activation_func,
        #     nn.Linear(NN_HIDDEN_SIZE, args.rnn_hidden_dim)
        # )

        # self.critiction = nn.CrossEntropyLoss()

        self.msg_net = nn.Sequential(
            nn.Linear(args.latent_dim + args.rnn_hidden_dim, NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_actions)
        )

        self.g_value_net = nn.Sequential(
            nn.Linear(self.latent_dim + args.rnn_hidden_dim, NN_HIDDEN_SIZE),
            activation_func,
            nn.Linear(NN_HIDDEN_SIZE, args.n_actions)
        )

        self.w_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.w_key = nn.Linear(args.latent_dim, args.attention_dim)
        # self.w_value = nn.Linear(args.attention_dim, args.attention_dim)

        self.tar_query = nn.Linear(args.rnn_hidden_dim, args.attention_dim)
        self.tar_key = nn.Linear(args.latent_dim, args.attention_dim)

    def init_hidden(self):
        # return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state, bs, test_mode=False, **kwargs):
        # x = F.relu(self.fc1(inputs))
        # print("the obs space is,", self.input_shape," the action space is ", self.n_actions)
        feature = self.input_qkv(inputs)
        k, q, v = th.chunk(feature, 3, dim=-1)
        q = q.unsqueeze(-2)
        k = k.unsqueeze(-1)
        feature_score = th.bmm(q, k) / (self.encode_dim ** (1 / 2))
        feature_weight = F.softmax(feature_score, dim=-1).view(-1, 1)
        feat = feature_weight * v
        '''
        q = self.q_proj(inputs)
        k = self.k_proj(inputs)
        v = self.v_proj(inputs)
        
        q1, q2 = th.chunk(q, 2, dim=-1)
        k1, k2 = th.chunk(k, 2, dim=-1)

        q1 = q1.unsqueeze(-2)
        q2 = q2.unsqueeze(-2)
        k1 = k1.unsqueeze(-1)
        k2 = k2.unsqueeze(-1)
        
        feature_score1 = th.bmm(q1, k1) / (self.encode_dim ** (1 / 2))
        feature_score2 = th.bmm(q2, k2) / (self.encode_dim ** (1 / 2))
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q1)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q2)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init

        feature_weight1 = F.softmax(feature_score1, dim=-1).view(-1, 1)
        feature_weight2 = F.softmax(feature_score2, dim=-1).view(-1, 1)
        feature_weight = feature_weight1 - lambda_full * feature_weight2 
        feat = feature_weight * v
        '''
        # activation_func = nn.LeakyReLU()
        # x = activation_func(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(feat, h_in)
        q_origin = self.fc2(h)

        # self_attention extract the feature
        # feature = self.input_qkv(h)
        # k, q, v = th.chunk(feature, 3, dim=-1)
        # q = q.unsqueeze(-2)
        # k = k.unsqueeze(-1)
        # feature_score = th.bmm(q,k)/(self.encode_dim**(1/2))
        # feature_weight = F.softmax(feature_score, dim=-1).view(-1,1)
        # feat = feature_weight * v

        h_repeat = h.view(bs, self.n_agents, -1).repeat(1, 1, self.n_agents).view(bs * self.n_agents * self.n_agents,
                                                                                  -1)
        # create tony_msg
        tiny_msg = self.tiny_msg_net(h.detach())
        # tiny_msg_repeat = tiny_msg.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        # calculate the infer demand
        infer_parameters = self.inference_net(tiny_msg)
        infer_parameters[:, -self.n_agents * self.latent_dim:] = th.clamp(
            th.exp(infer_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor)

        latent_infer = infer_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if test_mode:
            latent_i = latent_infer[:, :self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(latent_infer[:, :self.n_agents * self.latent_dim],
                                      (latent_infer[:, self.n_agents * self.latent_dim:]) ** (1 / 2))
            latent_i = gaussian_embed.rsample()  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent_i = latent_i.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)
        # latent_i = latent_i.reshape(bs, self.n_agents, self.n_agents, -1)
        # latent_i = latent_i.transpose(1, 2)
        # latent_i = latent_i.reshape(bs * self.n_agents * self.n_agents, -1)
        # based on the sample result, calculate the msg
        # tiny_msg_repeat = tiny_msg.view(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).view(bs * self.n_agents * self.n_agents, -1)
        # h_repeat = feat.view(bs, self.n_agents, -1).repeat(1, 1, self.n_agents).view(bs * self.n_agents * self.n_agents, -1)
        msg = self.msg_net(th.cat([h_repeat.detach(), latent_i], dim=-1)).view(bs, self.n_agents, self.n_agents,
                                                                               self.n_actions)
        # msg = self.msg_net(h_repeat).view(bs, self.n_agents, self.n_agents,self.n_actions)

        query = self.w_query(h.detach()).unsqueeze(1)
        key = self.w_key(latent_i)
        key = key.reshape(bs, self.n_agents, self.n_agents, -1)
        key = key.transpose(1, 2)
        key = key.reshape(bs * self.n_agents, self.n_agents, -1)
        key = key.transpose(1, 2)
        # key = self.w_key(latent_i).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        alpha = th.bmm(query / (self.args.attention_dim ** (1 / 2)), key).view(bs, self.n_agents, self.n_agents)
        for i in range(self.n_agents):
            alpha[:, i, i] = -1e9
        alpha = F.softmax(alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)

        # calculate the demand
        latent_parameters = self.demand_net(h)
        latent_parameters[:, -self.n_agents * self.latent_dim:] = th.clamp(
            th.exp(latent_parameters[:, -self.n_agents * self.latent_dim:]),
            min=self.args.var_floor)

        latent_embed = latent_parameters.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)

        if test_mode:
            latent = latent_embed[:, :self.n_agents * self.latent_dim]
        else:
            gaussian_embed = D.Normal(latent_embed[:, :self.n_agents * self.latent_dim],
                                      (latent_embed[:, self.n_agents * self.latent_dim:]) ** (1 / 2))
            latent = gaussian_embed.rsample()  # shape: (bs * self.n_agents, self.n_agents * self.latent_dim)
        latent = latent.reshape(bs * self.n_agents * self.n_agents, self.latent_dim)
        # latent = latent.reshape(bs, self.n_agents, self.n_agents, -1)
        # latent = latent.transpose(1, 2)
        # latent = latent.reshape(bs * self.n_agents * self.n_agents, -1)

        # h_g_repeat = feat.reshape(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).reshape(bs * self.n_agents * self.n_agents, -1)
        g_value = self.g_value_net(th.cat([h_repeat, latent], dim=-1)).view(bs, self.n_agents, self.n_agents,
                                                                            self.n_actions)
        # # g_value = self.g_value_net(h_g_repeat).view(bs, self.n_agents, self.n_agents,self.n_actions)
        #
        g_query = self.tar_query(h).unsqueeze(1)
        g_key = self.tar_key(latent)
        g_key = g_key.reshape(bs, self.n_agents, self.n_agents, -1)
        g_key = g_key.transpose(1, 2)
        g_key = g_key.reshape(bs * self.n_agents, self.n_agents, -1)
        g_key = g_key.transpose(1, 2)
        tar_alpha = th.bmm(g_query / (self.args.attention_dim ** (1 / 2)), g_key).view(bs, self.n_agents, self.n_agents)

        for i in range(self.n_agents):
            tar_alpha[:, i, i] = -1e9
        tar_alpha = F.softmax(tar_alpha, dim=-1).reshape(bs, self.n_agents, self.n_agents, 1)
        # tar_alpha[tar_alpha < (0.25 * 1 / self.n_agents)] = 0

        if test_mode:
            alpha[alpha < (0.3 * 1 / self.n_agents)] = 0

        gated_msg = alpha * msg

        return_q = q_origin.detach() + th.sum(gated_msg, dim=1).view(bs * self.n_agents, self.n_actions)
        return_g_q = q_origin + th.sum(tar_alpha * g_value, dim=1).view(bs * self.n_agents, self.n_actions)

        returns = {}
        if 'train_mode' in kwargs and kwargs['train_mode']:
            if hasattr(self.args, 'mi_loss_weight') and self.args.mi_loss_weight > 0:
                selected_action = th.max(return_g_q, dim=1)[1].unsqueeze(-1)
                one_hot_a = th.zeros(selected_action.shape[0], self.n_actions).to(self.args.device).scatter(1,
                                                                                                            selected_action,
                                                                                                            1)
                one_hot_a = one_hot_a.view(bs, 1, self.n_agents, -1).repeat(1, self.n_agents, 1, 1)
                one_hot_a = one_hot_a.view(bs * self.n_agents * self.n_agents, -1)
                # h_repeat = h_repeat.reshape(bs, self.n_agents, self.n_agents, -1)
                # h_repeat = h_repeat.transpose(1, 2)
                # h_repeat = h_repeat.reshape(bs * self.n_agents * self.n_agents, -1)

                demand_infer_embed = self.demand_inference_net(th.cat([h_repeat, one_hot_a], dim=-1))
                demand_infer_embed[:, -self.n_agents * self.latent_dim:] = th.clamp(
                    th.exp(demand_infer_embed[:, -self.n_agents * self.latent_dim:]),
                    min=self.args.var_floor)

                # demand_infer = demand_infer_embed.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)
                demand_infer = demand_infer_embed.reshape(bs, self.n_agents, self.n_agents, -1)
                demand_infer = demand_infer.transpose(1, 2)
                demand_infer = demand_infer.reshape(bs * self.n_agents, self.n_agents * self.latent_dim * 2)
                returns['demand_loss'] = self.calculate_action_mi_loss(bs, latent_embed, demand_infer) * 2
                returns['mi_loss'] = self.calculate_action_mi_loss(bs, latent_embed.detach(), latent_infer)

            if hasattr(self.args, 'entropy_loss_weight') and self.args.entropy_loss_weight > 0:
                query = self.tar_query(h.detach()).unsqueeze(1)
                key = self.tar_key(latent.detach())
                key = key.reshape(bs, self.n_agents, self.n_agents, -1)
                key = key.transpose(1, 2)
                key = key.reshape(bs * self.n_agents, self.n_agents, -1)
                key = key.transpose(1, 2)
                label_alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
                returns['entropy_loss'] = self.calculate_entropy_loss(label_alpha)
            # if hasattr(self.args, 'tiny_loss_weight'):
            #     returns['tiny_msg_loss'] = self.calculate_tiny_msg_loss(tiny_msg, h)
            # if hasattr(self.args, 'attention_loss_weight'):
            #     # query = self.w_query(h.detach()).unsqueeze(1)
            #     # key = self.w_key(tiny_msg_repeat.detach()).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
            #     # alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
            #     returns['attention_loss'] = self.calulate_attention_loss(tar_alpha, alpha)
            if hasattr(self.args, 'q_loss_weight'):
                returns['q_loss'] = self.calculate_q_loss(return_q, return_g_q.detach())

        if test_mode:
            return return_q, h, returns
        else:
            return return_g_q, h, returns

    def calulate_attention_loss(self, tar_alpha, alpha):
        # h_repeat = h.reshape(bs, self.n_agents, -1).repeat(1, self.n_agents, 1).reshape(bs * self.n_agents * self.n_agents, -1)
        # query = self.tar_query(h).unsqueeze(1)
        # key = self.tar_key(h_repeat).reshape(bs * self.n_agents, self.n_agents, -1).transpose(1, 2)
        # tar_alpha = F.softmax(th.bmm(query, key), dim=-1).reshape(bs, self.n_agents, self.n_agents)
        # alpha = alpha.reshape(bs,self.n_agents,-1)
        attention_loss = kl_divergence(Categorical(logits=tar_alpha), Categorical(logits=alpha)).sum(-1).mean()
        # attention_loss = kl_divergence(tar_alpha, alpha).sum(-1).mean()
        # attention_loss = self.critiction(tar_alpha, alpha)/bs
        return attention_loss * self.args.attention_loss_weight

    def calculate_q_loss(self, return_q, return_g_q):
        # q_max_info = th.max(return_g_q, dim=1)
        # selected_action = q_max_info[1].unsqueeze(-1)
        # new_q = q_max_info[0].unsqueeze(-1)
        # origin_q = return_q.gather(1, selected_action)
        delta_q = F.mse_loss(return_q, return_g_q)
        actor_loss = torch.mean(delta_q)
        return actor_loss * self.args.q_loss_weight
        # return actor_loss


    def calculate_tiny_msg_loss(self, tiny_msg, h):
        h_hat = self.decoder_net(tiny_msg)
        msg_loss = self.critiction(h_hat, h)
        return msg_loss

    def calculate_action_mi_loss(self, bs, latent_embed, latent_infer):
        latent_embed = latent_embed.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g1 = D.Normal(latent_embed[:, 0, :, :].reshape(-1, self.latent_dim),
                      latent_embed[:, 1, :, :].reshape(-1, self.latent_dim) ** (1 / 2))
        latent_infer = latent_infer.view(bs * self.n_agents, 2, self.n_agents, self.latent_dim)
        g2 = D.Normal(latent_infer[:, 0, :, :].reshape(-1, self.latent_dim),
                      latent_infer[:, 1, :, :].reshape(-1, self.latent_dim) ** (1 / 2))
        mi_loss = kl_divergence(g1, g2).sum(-1).mean()
        # return mi_loss
        return mi_loss * self.args.mi_loss_weight

    def calculate_entropy_loss(self, alpha):
        alpha = th.clamp(alpha, min=1e-4)
        entropy_loss = - (alpha * th.log2(alpha)).sum(-1).mean()
        return entropy_loss * self.args.entropy_loss_weight

