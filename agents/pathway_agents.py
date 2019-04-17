import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
import sys
import os
import math


class pathway_agents():

    def __init__(self, param):

        # agent variables
        self.rwd_prob = 1
        self.pun_prob = 1
        self.rwd_mag = param['rwd_mag']
        self.pun_mag = param['pun_mag']
        self.rho = param['rho']
        self.n_episodes = param['n_episodes']
        self.beta = param['beta']
        self.gamma = param['gamma']
        self.tsteps = param['tsteps']
        self.second_tone_list = param['second_tone_list']
        self.print_flag = param['print_flag']
        self.policy = param['policy']
        self.alpha_i = param['alpha_i']
        self.save_history = param['save_history']
        self.learning_rule =param['learning_rule']
        self.n_actions = 3
        self.n_pathways = 2
        self.n_test_episodes = 100

        # importing the environment
        self.get_environment()

    """
    ENVIRONMENT IMPORT
    This function imports the environment/task to be used later in training
    """
    def get_environment(self):

        # this function calls the timing task classical conditioning environment only

        cwd = os.getcwd()
        # changing the initial path given the current machine
        usr_path = ''
        for i in range(3):
            usr_path += str(cwd.split("/")[i]) + str('/')

        env_path = usr_path + 'Dropbox/phd/code/d1d2_project/classical_conditioning/'
        sys.path.append(env_path)

        from environments.timing_task_csc import timing_task_csc

        self.env = timing_task_csc(self.tsteps, self.second_tone_list,
                                   self.rwd_mag, self.rwd_prob, self.pun_mag, self.pun_prob)

    """
    TRANSFER FUNCTIONS
    These functions represent the transfer functions of the Reward Prediction Error used 
    """
    def tdp(self, x):
        if x > 0:
            return x
        else:
            return 0

    def tdn(self, x):
        if x < 0:
            return -x
        else:
            return 0

    def nl_tdp(self, x):
        sc = 6
        return sc*np.tanh(x)+sc*0.9

    def nl_tdn(self, x):
        sc = 5
        return sc*np.tanh(-x)+sc*0.5

    """
    AGENT POLICIES
    """
    def e_greedy(self, V, epsilon):

        # e-greedy
        if np.random.uniform(0, 1) > epsilon:
            action = np.random.choice(len(V))
        else:
            action = np.argmax(V)

        return int(action)

    def softmax(self, A, beta):
        """
        Altered version of the softmax function that takes into accout numerical overflow
        """

        axis = None

        x = A - A.max(axis=axis, keepdims=True)
        y = np.exp(x * beta)

        # avoiding numerical problems with the exponential - inifite values
        # take into account that this might affect the dynamic range of the calculation - TEST

        for i, e in enumerate(y):
            if np.isinf(e):
                y[i] = 1

        action_prob = y / y.sum(axis=axis, keepdims=True)
        selected_action = np.argmax(np.random.multinomial(1, action_prob, size=1))

        return selected_action

    """
    TRAINING ALGORITHMS
    """
    def train_agent(self):

        current_state = 0
        next_state = 0
        reward = 0
        action = 0

        # renaming environment
        env = self.env

        # learning rates for actor and critic - smaler alphas -> more iterations to converge
        alpha_v_cap = 0.01
        alpha_a_cap = 0.001
        alpha_v = np.ones(env.n_total_states)
        alpha_a = np.ones(env.n_total_states)

        # convergence measures
        pV = np.zeros(env.n_total_states)
        pA = np.zeros((self.n_pathways, env.n_total_states, self.n_actions))
        diffV = np.zeros(self.n_episodes)
        diffA = np.zeros((self.n_episodes, self.n_pathways, self.n_actions))

        # pathway weights - Anne Collins B_G and B_N respectively
        w_D = 1 + self.rho
        w_I = 1 - self.rho

        # td-error
        delta = 0

        # trial variables
        state_visits = np.zeros(env.n_total_states)

        # value functions for the simulations
        V = np.zeros(env.n_total_states)  # state value function
        # action value function
        Act = np.zeros((env.n_total_states, self.n_actions))
        A = np.random.uniform(
            0, 0.01, (self.n_pathways, env.n_total_states, self.n_actions))  # action weights

        # generating variables to record trial history
        if self.save_history:
            test_data = np.zeros((self.n_episodes, env.n_states, 6))
            V_h = np.zeros((self.n_episodes, env.n_total_states))
            A_h = np.zeros((self.n_episodes, self.n_pathways,
                            env.n_total_states, self.n_actions))
            delta_h = np.zeros((self.n_episodes, len(
                self.second_tone_list), self.tsteps))
            trial_seq = np.zeros((self.n_episodes, len(self.second_tone_list)))

        choices = np.zeros(self.n_test_episodes)

        for episode in range(self.n_episodes):

            trial_type = np.random.choice(
                np.arange(len(self.second_tone_list)))
            if self.save_history:
                trial_seq[episode, trial_type] = 1

            # comparing value functions
            pV = copy.copy(V)
            pA = copy.copy(A)

            current_state = 0

            for t in range(env.n_states):

                if type(self.policy) == int:
                    # action selection ------
                    if self.policy == 0:
                        action = env.opt_act[trial_type, current_state].astype(int)
                    elif self.policy == 1:
                        action = self.softmax(Act[current_state, :], self.beta)
                else:  # semi-optimal policy
                    action = self.policy[t].astype(int)

                # state update ------
                next_state, reward = env.get_outcome(
                    trial_type, current_state, action)

                # td update // reward has to be of the next_state ------
                delta = reward + self.gamma * V[next_state] - V[current_state]
                if self.save_history:
                    delta_h[episode, trial_type, t] = delta

                # calculating alpha -------
                state_visits[current_state] += 1
                inv_sv = self.alpha_i / state_visits[current_state]

                if inv_sv < alpha_v_cap:
                    alpha_v[current_state] = inv_sv
                else:
                    alpha_v[current_state] = alpha_v_cap

                if inv_sv < alpha_a_cap:
                    alpha_a[current_state] = inv_sv
                else:
                    alpha_a[current_state] = alpha_a_cap

                # update value functions ------ ANNE COLLINS & TRANSFER FUNCTION
                if self.learning_rule == 'anne_collins':
                    if current_state < env.n_total_states - 5:
                        # state value function updating
                        V[current_state] += alpha_v[current_state] * delta

                        # actor weights
                        A[0, current_state, action] += alpha_a[current_state] * A[0, current_state, action] * delta  # direct pathway
                        A[1, current_state, action] += alpha_a[current_state] * A[1, current_state, action] * (-delta)  # indirect pathway

                        # constrain actor values to be positive - heaviside function
                        A[0, current_state, action] = self.tdp(A[0, current_state, action])
                        A[1, current_state, action] = self.tdp(A[1, current_state, action])

                        # calculate actor values
                        Act[current_state, action] = w_D * A[0, current_state, action] - w_I * A[1, current_state, action]

                    else:  # clamp the final states to zero (terminal states)
                        V[current_state] = 0
                        A[0, current_state, action] = 0
                        A[1, current_state, action] = 0
                        Act[current_state, action] = 0

                if self.learning_rule == 'transfer_function':
                    if current_state < env.n_total_states - 5:
                        # state value function updating
                        V[current_state] += alpha_v[current_state] * delta

                        # advantage updating - mark humphries transfer function
                        A[0, current_state, action] += alpha_a[current_state] * \
                            (self.nl_tdp(delta) -
                             A[0, current_state, action])  # direct pathway
                        A[1, current_state, action] += alpha_a[current_state] * \
                            (self.nl_tdn(delta) -
                             A[1, current_state, action])  # indirect pathway

                        # calculate actor values
                        Act[current_state, action] = w_D * A[0, current_state,
                                                             action] - w_I * A[1, current_state, action]

                    else:  # clamp the final states to zero (terminal states)
                        V[current_state] = 0
                        A[0, current_state, action] = 0
                        A[1, current_state, action] = 0
                        Act[current_state, action] = 0

                # saving relevant variables ------
                if self.save_history:
                    V_h[episode, :] = V
                    A_h[episode, :] = A
                    test_data[episode, t, :] = np.asarray(
                        [trial_type, current_state, action, next_state, reward, delta])

                if self.print_flag and episode < 100:
                    print([trial_type, current_state,
                           action, next_state, reward, delta])

                # current state update
                current_state = next_state

                if current_state == env.n_total_states-5:

                    # calculating differences
                    diffV[episode] = np.sum((V-pV)**2)
                    for p in range(self.n_pathways):
                        for a in range(self.n_actions):
                            diffA[episode, p, a] = np.sum(
                                (A[p, :, a] - pA[p, :, a])**2)

                    # performance related variables
                    if episode > self.n_episodes - self.n_test_episodes:
                        if reward <= 0:
                            choices[self.n_episodes - episode -
                                    self.n_test_episodes] = 0
                        if reward > 0:
                            choices[self.n_episodes - episode -
                                    self.n_test_episodes] = 1

                    break

        if self.save_history:
            data_dict_hist = {'V': V, 'state_visits': state_visits, 'A': A, 'Act': Act,
                              'diffV': diffV, 'diffA': diffA, 'env': env, 'choices': choices,
                              'test_data': test_data, 'diffV': diffV, 'diffA': diffA, 'trial_seq': trial_seq,
                              'choices': choices, 'V_h': V_h, 'A_h': A_h, 'delta_h': delta_h}
            return data_dict_hist
        else:
            data_dict_no_hist = {'V': V, 'state_visits': state_visits, 'A': A, 'Act': Act,
                                 'diffV': diffV, 'diffA': diffA, 'choices': choices}
            return data_dict_no_hist
