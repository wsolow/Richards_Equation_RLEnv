"""
PPO LSTM Continuous Action Implemenation 

Based on CleanRL single file PPO implemenation

Written by Will Solow, 2025
"""
import torch.nn as nn
import torch.optim as optim
import torch
from torch.distributions.normal import Normal
import numpy as np
import os, time
from omegaconf import OmegaConf

from train_algs.base import setup_run, BaseTrainer

class PPO_LSTM_Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()

        self.rnn = nn.LSTM(np.array(envs.single_observation_space.shape).prod(), 128)
        self.hidden_layer = self.layer_init(nn.Linear(128, 128))
        
        self.h0 = nn.Parameter(torch.zeros(self.rnn.num_layers, envs.num_envs, self.rnn.hidden_size))
        self.c0 = nn.Parameter(torch.zeros(self.rnn.num_layers, envs.num_envs, self.rnn.hidden_size))

        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.actor_mean = self.layer_init(nn.Linear(128, np.array(envs.single_action_space.shape).prod()), std=0.01)
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.array(envs.single_action_space.shape).prod()))
        self.critic = self.layer_init(nn.Linear(128, 1), std=1)

    def get_states(self, x, lstm_state, done):
        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = x.reshape((-1, batch_size, self.rnn.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.rnn(
                h.unsqueeze(0),
                (
                    (1.0 - d).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        new_hidden = self.hidden_layer(new_hidden)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(hidden), lstm_state
    
    def get_action(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        action_mean = self.actor_mean(hidden)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        return probs.sample(), lstm_state
    
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

class PPO_Reccur(BaseTrainer):

    def __init__(self, config):

        super().__init__(config)

        self.config = config

        self.agent = PPO_LSTM_Agent(self.envs).to(self.device)

    def train(self):
        args = self.config.PPO
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size

        writer, run_name = setup_run(self.config)
        fpath = f"{os.getcwd()}{self.config.log_path}/{run_name}"
        with open(f"{fpath}/config.yaml", "w") as fp:
            OmegaConf.save(config=self.config, f=fp.name)
        fp.close()

        optimizer = optim.Adam(self.agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_observation_space.shape).to(self.device)
        actions = torch.zeros((args.num_steps, args.num_envs) + self.envs.single_action_space.shape).to(self.device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(self.device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(self.device)

        # Start environment
        global_step = 0
        start_time = time.time()
        next_obs, _ = self.envs.reset()
        next_obs = torch.tensor(next_obs).to(torch.float32).to(self.device)
        next_done = torch.zeros(args.num_envs).to(self.device)
        next_lstm_state = (self.agent.h0, self.agent.c0)

        for iteration in range(1, args.num_iterations + 1):
            initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # action logic
                with torch.no_grad():
                    action, logprob, _, value, next_lstm_state = self.agent.get_action_and_value(next_obs, next_lstm_state, next_done)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # Run environment
                next_obs, reward, terminations, truncations, infos = self.envs.step(np.clip(action.cpu().numpy(), 0, 0.05))
                next_done = np.logical_or(terminations, truncations)

                rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.tensor(next_obs).to(torch.float32).to(self.device), torch.tensor(next_done).to(self.device).to(torch.float32)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(
                    next_obs,
                    next_lstm_state,
                    next_done,
                ).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_dones = dones.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    _, newlogprob, entropy, newvalue, _ = self.agent.get_action_and_value(
                        b_obs[mb_inds],
                        (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            if iteration % args.log_frequency == 0:
                # Record rewards for plotting purposes
                writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
                writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
                writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
                writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
                writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
                writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
                writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
                writer.add_scalar("losses/explained_variance", explained_var, global_step)
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                torch.save(self.agent.state_dict(), f"{fpath}/ppo_agent.pt")

        torch.save(self.agent.state_dict(), f"{fpath}/ppo_agent.pth")

        writer.close()
