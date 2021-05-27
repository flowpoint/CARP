import gym
from ppo import BaseLinePPO
import torch

import util
import train
from constants import *

env = gym.make('CartPole-v0')
agent = BaseLinePPO(4, 2)
agent.cuda()

storage = util.RolloutStorage()

opt_V = torch.optim.AdamW(agent.critic.parameters(), lr = 1e-3)
opt_P = torch.optim.AdamW(agent.actor.parameters(), lr = 3e-4)

TIME_LIMIT = 9999
EPISODES = 1000

def prep_state(s):
    s = torch.from_numpy(s).float()
    s = s.to('cuda').unsqueeze(0)
    return s

for episode in range(EPISODES):
    agent.eval()
    s = env.reset()
    s = prep_state(s)
    total_r = 0

    for t in range(TIME_LIMIT):
        with torch.no_grad():
            pi, v, a = agent(s)
        v = v[0].item()
        log_prob = pi.log_prob(a)

        env.render()
        s_next, r, done, info = env.step(a.item())
        s_next = prep_state(s_next)
        total_r += r
        
        storage.remember(log_prob, v, s, a, r)
        
        if done: break

        s = s_next

    _, next_v, _ = agent(s_next)
    storage.set_terminal(next_v)

    loss = train.train_PPO(agent, opt_P, opt_V, storage)
    storage.reset()
    print(loss)
    print("REWARD: " + str(total_r))
env.close()
