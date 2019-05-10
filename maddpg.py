import pdb
import numpy as np
import time
from Network import *
import torch
from copy import *
from memory import *
from torch.autograd import *

class MADDPG():
    def __init__(self, n_agents, dim_obs, dim_act ):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents) ]
        self.critics = [Critic(n_agents, dim_obs, dim_act) for i in range(n_agents) ]
        self.n_agents = n_agents; self.n_states = dim_obs
        self.dim_act = dim_act

        self.memory = ReplayMemory()
        self.batch_size = 16
        self.episodes_before_train = 10

        self.GAMMA = 0.95; self.tau = 0.01
        self.scale_reward = 0.01
        self.lossfun = nn.MSELoss()

        self.var = [1.0 for i in range(n_agents) ]
        self.critic_optimizer = [torch.optim.Adam(x.parameters(), lr=0.001) for x in self.critics ]
        self.actor_optimizer = [torch.optim.Adam(x.parameters(), lr=0.0001) for x in self.actors ]

        self.episode_done = 0; self.steps_done = 0; self.load()

    def load(self):
        import os
        if os.path.exists("critic0.pth"):
            print("Load critics and actors net parameters.")
            for i in range(self.n_agents):
                self.critics[i].load_state_dict(torch.load("critic" + str(i) + ".pth") )
                self.actors[i].load_state_dict(torch.load("actor" + str(i) + ".pth") )

        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

    def select_action(self, states, actors):
        actions = Variable(torch.empty(self.n_agents, self.dim_act ) )
        for i in range(self.n_agents):
            state = states[i]
            actions[i] = actors[i](state )

        return actions

    def produce_action(self, states):
        actions = self.select_action(states, self.actors)
        for i in range(self.n_agents):
            np.random.seed(int(time.time()) )
            actions[i] += Variable(torch.tensor(np.random.randn(5) * self.var[i]).float() )

            if self.episode_done > self.episodes_before_train and self.var[i] > 0.05:
                self.var[i] *= 0.999998

            actions[i] = torch.clamp(actions[i], -1.0, +1.0)

        self.steps_done += 1
        return actions

    def train(self, i_episode):
        if self.episode_done < self.episodes_before_train:
            return
        for i in range(self.n_agents):
            transitions = self.memory.sample(self.batch_size );
            batch = Experience(*zip(*transitions) );
            non_final_mask = torch.ByteTensor(list(map(lambda s: s is not None, batch.next_states) ) )
            non_final_next_states = Variable(torch.Tensor( [s for s in batch.next_states if s is not None]) )
            rewards = Variable(torch.tensor(batch.rewards).float() )

            tmp = [self.actors_target[i](non_final_next_states[:, i, :]) for i in range(self.n_agents) ]
            non_final_next_actions = torch.stack(tmp)
            next_Q = torch.empty(self.batch_size );#pdb.set_trace()

            next_Q[non_final_mask] = self.critics_target[i](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1, self.n_agents * self.dim_act)
            ).squeeze()

            target_Q = next_Q * self.GAMMA + rewards[:,i].view(self.batch_size, -1) * self.scale_reward

            self.critic_optimizer[i].zero_grad(); #pdb.set_trace()
            whole_state = Variable(torch.Tensor(batch.states) ).view(self.batch_size, -1); #pdb.set_trace()
            whole_action = Variable(torch.stack(batch.actions) ).view(self.batch_size, -1)

            current_Q = self.critics[i](whole_state, whole_action)
            loss_Q = self.lossfun(current_Q, target_Q); #pdb.set_trace()
            loss_Q.backward()
            self.critic_optimizer[i].step()

            actor_loss = -self.critics[i](whole_state, whole_action, "actor") #pdb.set_trace()
            actor_loss = actor_loss.mean(); #pdb.set_trace()
            actor_loss.backward()
            self.actor_optimizer[i].step()

        def soft_update(target, source, t):
            for target_param, source_param in zip(target.parameters(),source.parameters() ):
                target_param.data.copy_( (1 - t) * target_param.data + t * source_param.data)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)
        if i_episode % 100 == 0 and i_episode > 0:
            print("Save critics and actors net parameters.")
            for i in range(self.n_agents):
                torch.save(self.critics[i].state_dict(), 'critic' + str(i) + ".pth" )
                torch.save(self.actors[i].state_dict(), 'actor' + str(i) + ".pth" )