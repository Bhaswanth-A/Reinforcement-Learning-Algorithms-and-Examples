import argparse
import os
from distutils.util import strtobool
import time
import random
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.distributions import MultivariateNormal


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"), help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="CartPole-v1", help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4, help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1, help='the seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=25000, help='total timesteps of the experiment') # Number of environment steps
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    
    parser.add_argument('--num-envs', type=int, default=4, help='the number of sub environments in the vector environment')
    parser.add_argument('--num-steps', type=int, default=128, help='the number of steps to run in each environment per policy rollout') # How much data to collect per rollout
    # For each policy rollout, we collect 4*128=512 data points for training. This is known as the batch size
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='toggle learning rate annealing for policy and value networks')
    parser.add_argument('--gae', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='use GAE for advantage computation')
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95, help='lambda for general advantage estimation')
    parser.add_argument('--num-minibatches', type=int, default=4, help='number of minibatches')
    parser.add_argument('--update-epochs', type=int, default=4, help='K epochs to update the policy')
    parser.add_argument('--clip-coef', type=float, default=0.2, help='the surrogate clipping coefficient')
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='toggles whether or not to use a clipped loss for the value function, as per the paper')
    parser.add_argument('--ent-coef', type=float, default=0.01, help='coefficient of entropy')
    parser.add_argument('--vf-coef', type=float, default=0.5, help='coefficient of value function')
    parser.add_argument('--max-grad-norm', type=float, default=0.5, help='maximum norm of the gradient clipping')

    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)

    return args

if __name__=='__main__':
    args = parse_args()
    print(args)
    run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"   # Setup unique run name for experiment
    writer = SummaryWriter(f"runs/{run_name}") # Save metrics to a folder

    # Setup random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")


    def make_env(gym_id, seed, idx, capture_video, run_name):
        def thunk():
            env = gym.make(gym_id, render_mode="rgb_array")
            env = gym.wrappers.RecordEpisodeStatistics(env)
            if capture_video:
                if idx == 0:
                    env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env
        return thunk

    # Make environment
    # env = gym.make(args.gym_id, render_mode="rgb_array")
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.RecordVideo(env, "videos")
    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    observation = envs.reset()
    episodic_return = 0
    for _ in range(200):
        action = envs.action_space.sample()
        observation, reward, terminated, truncated, info = envs.step(action)
        # print(info)
        episodic_return += reward
        if terminated.any():
            observation = envs.reset()
            episodic_return = 0
    envs.close()
    print(envs.single_action_space.n)
    print(envs.single_observation_space.shape)

    
    # envs = gym.vector.SyncVectorEnv([make_env(args.gym_id)])
    # observation = envs.reset()
    # for _ in range(200):
    #     action = envs.action_space.sample()
    #     observation, reward, terminated, truncated, info = envs.step(action)
    #     print(info)
    #     # for item in info:
    #     #     if "episode" in item.keys():
    #     if terminated:
    #         print(f"episodic return {info['episode']['r']}")


    class Agent(nn.Module):
        def __init__(self, env):
            super(Agent, self).__init__()

            self.cov_var = torch.full(size=(envs.single_action_space.n,), fill_value=0.5)
            self.cov_mat = torch.diag(self.cov_var)

            # Setup critic network
            self.critic = nn.Sequential(
                self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                self.layer_init(nn.Linear(64,64)),
                nn.Tanh(),
                self.layer_init(nn.Linear(64,1), std=1.0)
            )

            # Setup actor network
            self.actor = nn.Sequential(
                self.layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                self.layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                self.layer_init(nn.Linear(64, np.array(envs.single_action_space.n)), std=0.01)
            )


        def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0): # Pytorch uses different layer initializations. PPO uses different
            torch.nn.init.orthogonal_(layer.weight, std)
            torch.nn.init.constant_(layer.bias, bias_const)
            return layer
        
        def get_value(self, obs):
            return self.critic(obs)
        
        def get_action_and_value(self, obs, action=None):
            logits = self.actor(obs)
            # print(logits.shape)
            probs = Categorical(logits=logits) # Softmax operation to get action probability distribution
            if action is None:
                action = probs.sample()
                # print(action)
            return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

            # Method used in Implementation 1
            # dist = MultivariateNormal(logits, self.cov_mat)
            # action = dist.sample()
            # return action, dist.log_prob(action), self.critic(obs)


    agent = Agent(envs)
    # print(agent)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # Rollout storage setup - capturing 512 data points
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    # print(envs.reset())

    global_step = 0 # Track number of environment steps
    start_time = time.time() # Helps calculate frames per second
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    # print(next_obs.shape)     # Each environment has 4 observations. Hence torch.Size([4, 4])
    next_done = torch.zeros(args.num_envs).to(device)
    # print(next_done. shape)   # Done flag for each environment. Hence torch.Size([4])
    num_updates = args.total_timesteps // args.batch_size # Num of updates = 25000/512 = 48
    # print(num_updates)
    # print("next_obs.shape", next_obs.shape)
    # print("agent.get_value(next_obs)", agent.get_value(next_obs)) # We get 4 scalar values for each environment
    # print("agent.get_value(next_obs).shape", agent.get_value(next_obs).shape)
    # print("\nagent.get_action_and_value(next_obs)", agent.get_action_and_value(next_obs))


    for update in range(1, num_updates+1):
        # Annealing the learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow     # Updating the learning rate

        # Policy rollout
        for step in range(args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done 

            with torch.no_grad():   # No need to cache gradients during rollout
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # print(action.shape, actions.shape)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(np.array(action.unsqueeze(dim=0)[0])) # Input should be numpy array
            # done = (terminated or truncated).all()
            done = np.logical_or(terminated, truncated)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs).to(device)
            next_done = torch.Tensor(done).to(device)

            for item in info:
                if "final_info" in item:
                    for i,j in enumerate(info['final_info']):
                        if j!=None:
                            print(f"global_step={global_step}, episodic_return={info['final_info'][i]['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", info['final_info'][i]["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info['final_info'][i]["episode"]["l"], global_step)
                            break

            
            # GAE standard implementation
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1,-1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    lastgaelam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            nextvalues = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t+1]
                            nextvalues = values[t+1]
                        delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                        advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            nextnonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            nextnonterminal = 1.0 - dones[t+1]
                            next_return = returns[t+1]
                        returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return
                    advantages = returns - values

            # Flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, new_values = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    mb_advantages = b_advantages[mb_inds]
                    
                    # Advantage normalization
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1-args.clip_coef, 1+args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = new_values.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds])**2).mean()

                    entropy_loss = entropy.mean() # Entropy is a measure of the level of chaos in action probability distribution
                    # Intutitively, maximizing entropy would encourage the agent to explore more
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm) # Extra line
                    optimizer.step()
                    
                    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                    # print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    envs.close()
    writer.close()

                
