import gym
from ppo import BaseLinePPO
import torch

import util
import train
from constants import *

env = gym.make('CartPole-v1')
agent = BaseLinePPO(4, 512, 2)
agent.cuda()

storage = util.RolloutStorage()

opt = torch.optim.AdamW(agent.parameters(), lr = LEARNING_RATE)

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
            pi, v = agent(s)
        a = agent.sample_from_probs(pi).long()
        v = v[0].detach()

        log_prob = torch.log(pi)[0,a]

        env.render()
        s_next, r, done, info = env.step(a.item())
        s_next = prep_state(s_next)
        total_r += r

        done = 1 if done else 0
        m = util.makeMask(done)
        r = torch.tensor([r], device = 'cuda').float()
        
        storage.add([log_prob, v, s, a, r, m])
        
        if done: break

        s = s_next

    _, next_v = agent(s_next)
    #storage.update_gae(next_v)

    loss = train.train_PPO(agent, opt, storage)
    storage.reset()
    print(loss)
    print("REWARD: " + str(total_r))
env.close()
    



