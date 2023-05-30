# Refer to pseudo code

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from network import FeedFowrwardNN
from torch.optim import Adam
import numpy as np
import gym

class PPO:

    def __init__(self, env):

        self._init_hyperparameters()

        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        # ALGORITHM Step 1
        # Initialize actor and critic networks
        self.actor = FeedFowrwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedFowrwardNN(self.obs_dim, 1)

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)

        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

    
    def get_action(self, obs):

        # Query the actor network for a mean action
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(np.array(list(obs[0]), dtype=np.float32))

        # Create multivariate normal distribution
        dist = MultivariateNormal(mean, self.cov_mat)

        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach() # Detaches tensor from computation graph


    def _init_hyperparameters(self):
        self.timesteps_per_batch = 4800
        self.max_timesteps_per_episode = 1600
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2     # As recommended by the paper
        self.lr = 0.005     # learning rate


    # ALGORITHM STEP 3
    def rollout(self):
        # Batch data - we collect data from a set of episodes by running the actor policy
        batch_obs = []
        batch_acts = []
        batch_log_probs = []
        batch_rews = []
        batch_rtgs = [] # batch rewards to-go
        batch_lens = [] # episodic lengths in batch

        # In our batch, we’ll be running episodes until we hit self.timesteps_per_batch 
        # timesteps; in the process, we shall collect observations, actions, log probabilities 
        # of those actions, rewards, rewards-to-go, and lengths of each episode.

        # Number of timesteps run so far
        t = 0

        while t <  self.timesteps_per_batch:

            ep_rews = []

            obs = self.env.reset()
            done = False

            for ep_t in range(self.max_timesteps_per_episode):
                t += 1

                batch_obs.append(obs) # Collect observation

                action, log_prob = self.get_action(obs)

                obs, rew, done, _, _ = self.env.step(action)

                ep_rews.append(rew)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)

                if done:
                    break
            
            # Collect episodic lengths and rewards
            batch_lens.append(ep_t+1)
            batch_rews.append(ep_rews)

        # Reshape data to tensors
        batch_obs = torch.tensor(batch_obs, dtype=torch.float)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)


        # ALGORITHM STEP 4
        batch_rtgs = self.compute_rtgs(batch_rews)

        return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens
    

    # ALGORITHM STEP 4
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        
        # Iterate through each episode backwards to maintain same order in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)

        return batch_rtgs


    def learn(self, total_timesteps):

        t_so_far = 0 # Timesteps simulated so far
        while t_so_far < total_timesteps:   # ALGORITHM Step 2
            # ALGORITHM STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens = self.rollout()

            # Calculate how many time steps we collected this batch
            t_so_far += np.sum(batch_lens)

        # Calculate V_{phi,k}
        V, _ = self.evaluate(batch_obs, batch_acts)
        
        # ALGORITHM Step 5
        # Calculate advantage function
        A_k = batch_rtgs - V.detach() # we do V.detach() since V is a tensor with gradient required. 
                                    # However, the advantage will need to be reused each epoch loop, and the computation graph 
                                    # associated with advantage at the k-th iteration will not be useful in multiple epochs of 
                                    # stochastic gradient ascent.

        # Normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10) # 1e-10 is added to avoid possibility of dividing by zero


        for _ in range(self.n_updates_per_iteration):
            # Calculate pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

            # Calculate ratio
            ratios = torch.exp(curr_log_probs - batch_log_probs)    # since both batch_log_probs and curr_log_probs are log probs,
                                                                    # we can just subtract them and exponentiate the log out with e


            # Calculate surrogate losses
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            actor_loss = (-torch.min(surr1, surr2)).mean()      # We will use Adam optimizer. So minimizing the negative loss
                                                                # maximizes the performance function

            critic_loss = nn.MSELoss()(V, batch_rtgs)

            
            # Calculate gradients and perform back propagation for actor network
            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            # Calculate gradients and perform back propagation for critic network
            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()


    def evaluate(self, batch_obs, batch_acts):

        # Query critic network for a value V for each obs in batch_obs
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using the most recent actor network
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)

        return V, log_probs



env = gym.make('Pendulum-v1')
model = PPO(env)
model.learn(10000)