from MADDPG import MADDPG
import numpy as np
import torch as th
from params import scale_reward
from envs import matrixgame
from scipy.special import softmax
from torch.autograd import Variable

e_render = False

world = matrixgame.MatrixGame()
n_coop = 2

reward_record = []

np.random.seed(1234)
th.manual_seed(1234)
n_agents = 2#world.n_pursuers
n_states = 1
n_actions = 2
capacity = 1000000
batch_size = 32

n_episode = 3001
max_steps = 60
episodes_before_train = 10


win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        # render every 100 episodes to speed up training
        obs = obs.type(FloatTensor)
        action = maddpg.select_action(obs).data.cpu()
        if t%4==0 and i_episode==11:
            pass
            #print("action...",action)
            #print("....",maddpg.actors[0](Variable(th.Tensor( [[0]]))))
        #actli = [np.argmax(x) for x in action.numpy()]
        actli = [np.random.choice(2,1,p = softmax(pr))[0] for pr in action.numpy()]
        obs_, reward, done, _ = world.step(actli)
        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, action, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    
    if (i_episode)%10==0 and i_episode!=0 and i_episode!=10:
        for k in range(2):
            for j in range(2):
                sta = Variable(th.Tensor( [[0,0]]))
                acts =  Variable(th.Tensor( [[k,1-k,j,1-j]]))
                Q = maddpg.critics[0](sta,acts)
                print("i..j..",k,j,"..Q..",Q)
        print("closs..",[c_loss[0].detach().numpy(),c_loss[1].detach().numpy()],"..aloss..",[a_loss[0].detach().numpy(),a_loss[1].detach().numpy()])

    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

