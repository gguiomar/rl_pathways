import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy

class timing_task_csc():
    
    # things to implement:
    # 1. automatic splitting of trial identities (left,right) for any number of second tones
    
    def __init__(self, n_states, second_tone_list, reward_magnitude, 
                 reward_probability, punishment_magnitude, punishment_probability):
        
        # Task variables
        self.n_states = n_states # separate total number of transitions from total number of states
        self.n_total_states = 2 * n_states + 5
        self.n_actions = 3
        
        self.action_dict = np.arange(self.n_actions)
        self.second_tone_list = second_tone_list
        self.trial_types = len(self.second_tone_list)
        self.n_trials = len(self.second_tone_list)
        self.trials = np.zeros((self.n_trials, self.n_states*2, self.n_states))

        
        # Reward variables
        self.reward_state = [0,0]
        self.reward_magnitude = reward_magnitude
        self.punishment_magnitude = punishment_magnitude
        self.reward_probability = reward_probability
        self.punishment_probability = punishment_probability

        self.generate_environment()

    
    """
    GENERATORS
    These functions generate the necessary ingredients to define the Markov Decision Process (MDP)
    of the interval timing categorisation. [PUT REFERENCE]

    ---- Short description of the task ---
    ...
    """
    def generate_environment(self):
        """
        Main generating function: Takes the state representation generates the state sequence.
        """
        self.generate_state_representation()
        self.generate_state_sequence()
        self.generate_environment_solution(0)
    
    def generate_state_representation(self):
        
        self.n_trials = len(self.second_tone_list)
        tone1 = np.zeros((self.n_trials, self.n_states, self.n_states))
        tone2 = np.zeros((self.n_trials, self.n_states, self.n_states))
        self.trials = np.zeros((self.n_trials, self.n_states*2, self.n_states))

        for k,e in enumerate(self.second_tone_list):
            for i in range(self.n_states):
                if i > self.n_states-5:
                    tone1[k,i,i] = 0
                    tone2[k,i,i] = 1
                elif i < e:
                    tone1[k,i,i] = 1
                    tone2[k,i,i] = 0
                else:
                    tone1[k,i,i] = 0
                    tone2[k,i,i] = 1
                    
            #self.trials.append(np.concatenate((tone1[k,:,:], tone2[k,:,:]), axis = 0))
            self.trials[k,0:self.n_states,:] = tone1[k,:,:]
            self.trials[k,self.n_states: 2*self.n_states,:] = tone2[k,:,:]
        
    def generate_state_sequence(self):
        
        # this function generates the state identities for the MDP
        # that will be used in the simulation
        
        state = 0
        self.trial_st = np.zeros((self.n_trials, self.n_states))
        for k in range(self.n_trials):
            cnt = 0
            base = 0
            for i,e in enumerate(self.trials[k].T):
                state = np.nonzero(e)[0]
                if i == 0:
                    self.trial_st[k,i] = 0
                if i != 0 and state != 0:
                    self.trial_st[k,i] = state
                if not state and i != 0:
                    self.trial_st[k,i] = base + cnt
                    cnt += 1
        
        self.trial_st = self.trial_st.astype(int)
    
    def generate_environment_solution(self, plot_flag):

        # generates the policy for the optimal agent

        # map second tone to state identity
        self.second_tone_state = np.zeros(self.n_trials)
        for tr, st in enumerate(self.second_tone_list):
            self.second_tone_state[tr] = np.nonzero(
                self.trials[tr][:, st])[0][0]

        # map state identity to optimal action
        self.opt_act = 2 * np.ones((self.n_trials, self.n_total_states))
        for tr in range(self.n_trials):
            for i, state in enumerate(self.trial_st[tr]):
                if state >= self.second_tone_state[tr]:
                    # split decisions in the middle
                    if tr < self.n_trials/2:  # short decision
                        self.opt_act[tr, state] = 0
                    else:  # long decision
                        self.opt_act[tr, state] = 1

        # take into account that the opt_act vector will have a set of 2's
        # in the end because of the 0 states after the episode finishes

        if plot_flag:
            plt.figure(figsize=(20, 10))
            plt.title('Optimal actions')
            plt.imshow(self.opt_act)
            plt.colorbar(fraction=0.01)
            plt.ylabel('trial type')
            plt.xlabel('state')
            plt.show()

    """
    VISUALIZATIONS
    These functions show human readable representations of the M
    """
    def plot_state_representation(self):
        
        fig = plt.figure(figsize=(15,10))
        gs = gridspec.GridSpec(1, len(self.second_tone_list))
        ax = []

        #titles = ['short 1', 'short 2', 'long 1', 'long 2']
        for k,e in enumerate(self.second_tone_list):
            ax.append([])
            ax[k] = fig.add_subplot(gs[0,k])
            if k == 0: 
                ax[k].set_ylabel('State')
            ax[k].set_xlabel('Time')
            ax[k].imshow(self.trials[k], cmap='Greys')
            #ax[k].set_title(titles[k])
            
        plt.show()
    
    """
    TESTING THE ENVIRONMENT
    These functions go through all states and actions in order to check for
    weird state,action transtions
    """        
    def test_environment_all(self):
        
        for tt in range(self.trial_types):
            for cs in self.trial_st[tt]:
                for a in range(self.n_actions):
                    print([tt, cs, a], self.get_outcome(tt, cs, a))
                    
    def test_environment_action(self, action):
        current_state = 0
        next_state = 0
        for i,tr in enumerate(second_tone_list):
            for j,current_state in enumerate(env.trial_st[i]):
                next_state, reward = env.get_outcome(i, current_state, action)
                print([i, current_state, action, next_state, reward])

    """
    GET OUTCOME
    Main function of the enviromment; defines the MDP
    """
    def get_outcome(self, current_trial, current_state, action):
        
        next_state = 0
        reward = 0
        check_valid_state = np.argwhere(self.trial_st[current_trial] == current_state).shape[0] # should be > 0
            
        if check_valid_state: 

            # making a choice and going to terminal state
            if action != 2:
                state_index = np.argwhere(self.trial_st[current_trial] == current_state)[0][0]
                
                # if a decision is made before the terminal states
                if current_state < self.n_total_states - 5: # buffer states 
                    
                    if action == self.opt_act[current_trial, current_state]:
                        reward = self.reward_magnitude
                        next_state = self.n_total_states - 5 # transition into terminal states
                    else:                                                       
                        reward = self.punishment_magnitude
                        next_state = self.n_total_states - 5 # transition into terminal states
                
                # when we are in the terminal states
                elif self.n_total_states - 5 <= current_state < self.n_total_states - 1:
                    reward = 0
                    next_state = 0
                
                # when we reach the final terminal state we go back to the initial state
                elif current_state == self.n_total_states:
                    reward = 0
                    next_state = 0
 
            # hold action and moving along the state space
            else:
                state_index = np.argwhere(self.trial_st[current_trial] == current_state)[0][0] #current state index

                if state_index < np.argmax(self.trial_st[current_trial]):
                    next_state = self.trial_st[current_trial][state_index + 1]
                else:
                    next_state = 0
                    reward = 0
        else:
            next_state = 0
            reward = 0

        
        return next_state, reward
