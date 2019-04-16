# PLOTTING AND ANALYSIS FUNCTIONS
from matplotlib import rc
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import matplotlib.gridspec as gridspec
import copy as copy
import scipy as sp
from scipy import ndimage

class pathway_analysis():

    """
    This class's main purpose is to generate useful visualiations of the data generated by the artificial agents.
    It takes as input a data structure called "sim_data", from which it should be able to get all necessary
    information.
    """

    def __init__(self, sim_data):
        self.sim_data = sim_data
        self.sim_n = len(sim_data) # number of simulations in this data structure

    """
    USEFUL FUNCTIONS
    These are basic functions used in calculating agent performance measures and other simple transformations
    on vectors that are recurrently used in the code.
    """
    
    def calc_perf(self):
        """Calculates the performance of the agent on a single experiment."""
        sim_data = self.sim_data
        choices = sim_data['choices']
        return sum(choices)/choices.shape[0]
    
    def normalise_vec(self, vec):
        """normalises a vector"""
        return (vec-min(vec))/(max(vec)-min(vec))
    
    """
    SIMULATION DATA MANIPULATION FUNCTIONS
    These functions take the sim_data structure and change it into a useful shape for plotting
    """

    def avg_sims_pathways(self):
        """
        Averages the data from multiple simulations - Is called by grid plot
        """
        

        sim_data = self.sim_data
        # averaging state value function
        sim_n = len(sim_data)
        allV = np.zeros((sim_n,sim_data[0]['V'].shape[0]))

        for i in range(sim_n):
            allV[i,:] = sim_data[i]['V']
        avgV = np.mean(allV, axis = 0)
        varV = np.var(allV, axis = 0)

        # averaging action value function
        n_actions = 3
        n_pathways = 2
        allA = np.zeros((n_pathways, n_actions, sim_n, sim_data[0]['A'].shape[1]))
        avgA = np.zeros((n_pathways, n_actions, sim_data[0]['A'].shape[1]))
        varA = np.zeros((n_pathways, n_actions, sim_data[0]['A'].shape[1]))

        for p in range(n_pathways):
            for a in range(n_actions):
                for i in range(sim_n):
                    allA[p,a,i,:] = sim_data[i]['A'][p,:,a]
                avgA[p,a,:] = np.mean(allA[p,a,:,:], axis = 0)
                varA[p,a,:] = np.var(allA[p,a,:,:], axis = 0)

        # averaging diffV and diffA - filter the signals
        allDiffV = np.zeros((sim_n,sim_data[0]['diffV'].shape[0]))

        for i in range(sim_n):
            allDiffV[i,:] = sp.ndimage.filters.gaussian_filter1d(sim_data[i]['diffV'], 500)
        avgDiffV = np.mean(allDiffV, axis = 0)
        varDiffV = np.var(allDiffV, axis = 0)

        allDiffA = np.zeros((n_pathways, n_actions, sim_n,sim_data[0]['diffA'].shape[0]))
        avgDiffA = np.zeros((n_pathways, n_actions, sim_data[0]['diffA'].shape[0]))
        varDiffA = np.zeros((n_pathways, n_actions, sim_data[0]['diffA'].shape[0]))

        for p in range(n_pathways):
            for a in range(n_actions):
                for i in range(sim_n):
                    allDiffA[p,a,i,:] = sp.ndimage.filters.gaussian_filter1d(sim_data[i]['diffA'][:,p,a], 500)
                avgDiffA[p,a,:] = np.mean(allDiffA[p,a,:,:], axis = 0)
                varDiffA[p,a,:] = np.var(allDiffA[p,a,:,:], axis = 0)
        
        avg_data_dict = {'avgV':avgV, 'avgA':avgA, 'varV':varV, 'varA':varA, 'avgDiffV':avgDiffV, 
                        'varDiffV':varDiffV, 'avgDiffA':avgDiffA, 'varDiffA':varDiffA}
            
        return avg_data_dict

    def avg_value_functions(self):
        # averaging state value function

        sim_data = self.sim_data

        allV = np.zeros((self.sim_n, sim_data[0][0].shape[0]))
        for i in range(self.sim_n):
            allV[i, :] = sim_data[i][0]
        avgV = np.mean(allV, axis=0)

        # averaging action value function
        actions = 3
        allA = np.zeros((actions, self.sim_n, sim_data[0][0].shape[0]))
        avgA = np.zeros((actions, sim_data[0][0].shape[0]))
        for a in range(3):
            for i in range(self.sim_n):
                allA[a, i, :] = sim_data[i][2][:, a]
            avgA[a, :] = np.mean(allA[a, :, :], axis=0)

        return avgV, avgA

    """
    PLOTTING FUNCTIONS
    These functions take the sim_data structure or its transformations and generate visualizations regarding: 
    1. Behaviour of the agents
    2. Value functions and their evolution
    3. Convergence properties of the algorithms
    """
    def plot_sims(self):

        sim_data = self.sim_data

        print('State Value Function')
        plt.plot(sim_data[0])
        plt.show()

        print('Advantage Functions:')
        plt.plot(sim_data[2][:, 0], 'b', label='0')
        plt.plot(sim_data[2][:, 1], 'r', label='1')
        plt.plot(sim_data[2][:, 2], 'k', label='2')
        plt.legend()
        plt.show()

        print('State Value function convergence')
        plt.plot(sim_data[4])
        plt.show()

        print('State Visits')
        plt.plot(sim_data[1])
        plt.show()

        # change this function's name to something more general as it's used to plot almost all information regarding states

    def grid_plot_pathway_sim(self):

        sim_data = self.sim_data

        V, state_visits, A, diffV, diffA = sim_data['V'], sim_data[
            'state_visits'], sim_data['A'], sim_data['diffV'], sim_data['diffA']
        sim_perf = calc_perf(sim_data)

        # create the figure object
        columns = 4
        rows = 2

        fig = plt.figure(figsize=(25, 30))
        gs = gridspec.GridSpec(columns, rows)

        ax = []
        for c in range(columns):
            ax.append([])
            for r in range(rows):
                ax[c].append(fig.add_subplot(gs[c, r]))

        ax[0][0].plot(V, 'k')
        ax[0][0].set_title('State Value Function - Perf: ' + str(sim_perf))
        ax[0][0].set_xlabel('State')
        ax[0][0].set_ylabel('V(S)')

        ax[0][1].plot(A[0, :, 2], 'b', label='D1')
        ax[0][1].plot(A[1, :, 2], 'r', label='A2A')
        ax[0][1].legend()
        ax[0][1].set_title('Hold Action Advantage Function')
        ax[0][1].set_xlabel('State')
        ax[0][1].set_ylabel('A(S,Hold)')

        ax[1][0].plot(A[0, :, 0], 'b', label='D1')
        ax[1][0].plot(A[1, :, 0], 'r', label='A2A')
        ax[1][0].legend()
        ax[1][0].set_title('Left Action Advantage Function')
        ax[1][0].set_xlabel('State')
        ax[1][0].set_ylabel('A(S,Left)')

        ax[1][1].plot(A[0, :, 1], 'b', label='D1')
        ax[1][1].plot(A[1, :, 1], 'r', label='A2A')
        ax[1][1].legend()
        ax[1][1].set_title('Right Action Advantage Function')
        ax[1][1].set_xlabel('State')
        ax[1][1].set_ylabel('A(S,Right)')

        ax[2][0].plot(state_visits, 'k')
        ax[2][0].set_title('State Visits: ')
        ax[2][0].set_xlabel('State')
        ax[2][0].set_ylabel('Visits')

        dV = sim_data['diffV']
        dA = sim_data['diffA']
        filtered_dV = sp.ndimage.filters.gaussian_filter1d(dV, 500)
        filtered_dA = np.zeros(dA.shape)

        ax[2][1].plot(normalise_vec(filtered_dV), 'k')
        ax[2][1].set_title('State Visits: ' + str(sim_perf))
        ax[2][1].set_xlabel('episode')
        ax[2][1].set_ylabel('$V(s_t)_k - V(s_t)_{k-1}$')

        for p in range(2):
            for a in range(3):
                filtered_dA[:, p, a] = sp.ndimage.filters.gaussian_filter1d(
                    dA[:, p, a], 500)

        for a in range(3):
            ax[3][0].plot(normalise_vec(filtered_dA[:, 0, a]), label=str(a))

        ax[3][0].set_title('Convergence - Direct Pathway')
        ax[3][0].set_xlabel('episode')
        ax[3][0].set_ylabel(r'$A^p_{k+1}(t) - A^p_k(t)$')

        for a in range(3):
            ax[3][1].plot(normalise_vec(filtered_dA[:, 1, a]), label=str(a))

        ax[3][1].set_title('Convergence - Indirect Pathway')
        ax[3][1].set_xlabel('episode')
        ax[3][1].set_ylabel(r'$A^p_{k+1}(t) - A^p_k(t)$')

        return fig

    def grid_plot_learning_hist(self, only_show):

        sim_data = self.sim_data

        V_h = sim_data['V_h']
        A_h = sim_data['A_h']

        columns = 3
        rows = 3

        fig = plt.figure(figsize=(20, 10))
        gs = gridspec.GridSpec(columns, rows)

        ax = []
        for c in range(rows):
            ax.append([])
            for r in range(columns):
                if c == 0:
                    ax[c].append(fig.add_subplot(gs[c, :]))
                else:
                    ax[c].append(fig.add_subplot(gs[c, r]))

        # state value function
        for i, e in enumerate(V_h):
            if i % only_show == 0:
                ax[0][0].plot(e)

        # direct pathway
        for a in range(3):
            for i, e in enumerate(A_h):
                if i % only_show == 0:
                    ax[1][a].plot(e[0, :, a])

        # indirect pathway
        for a in range(3):
            for i, e in enumerate(A_h):
                if i % only_show == 0:
                    ax[2][a].plot(e[1, :, a])

        return fig

    def plot_bulk_data(self):

        sim_data = self.sim_data

        A = sim_data['A']
        fig = plt.figure(figsize=(10, 5))
        plt.plot(np.sum(A[0, :, :], axis=1), 'b', label='$A^D$')
        plt.plot(np.sum(A[1, :, :], axis=1), 'r', label='$A^I$')
        plt.xlabel('State')
        plt.ylabel('$A^p$')
        plt.legend()

        return fig

    def generate_grid_plot_pathways(self):
        sim_data = self.sim_data

        columns = 2
        rows = 3

        fig = plt.figure(figsize=(30,25))
        gs = gridspec.GridSpec(rows, columns)
        
        # average the data from all the simulations
        avgData = avg_sims_pathways(sim_data)

        avgV = avgData['avgV']
        varV = avgData['varV']
        avgA = avgData['avgA']
        varA = avgData['varA']
        avgDiffV = avgData['avgDiffV']
        varDiffV = avgData['varDiffV']
        avgDiffA = avgData['avgDiffA']
        varDiffA = avgData['varDiffA']
        
        # make this an array dependent variable
        n_pathways = 2
        n_actions = 3

        x1 = np.arange(avgData['varV'].shape[0]).astype(int)
        x2 = np.arange(avgData['avgDiffV'].shape[0]).astype(int)

        colors = ['blue','green','red']
        labels = ['L', 'R', 'H']
        
        ax = []
        for c in range(rows):
            ax.append([])
            for r in range(columns):
                ax[c].append(fig.add_subplot(gs[c,r]))

        ax[0][0].plot(x1, avgV, 'k', linewidth = 1)
        ax[0][0].fill_between(x1, avgV - varV, avgV + varV, color = 'gray', alpha = 0.2)
        ax[0][0].set_ylabel('$V(s_t)$')
        ax[0][0].set_xlabel('$s_t$')
        ax[0][0].set_title('State value function')  
        ax[0][1].plot(x2, avgDiffV, 'k', linewidth = 1)
        ax[0][1].fill_between(x2, avgDiffV - varDiffV, avgDiffV + varDiffV, color = 'gray', alpha = 0.2)
        ax[0][1].set_title('State value function convergence')  
        ax[0][1].set_ylabel('$V_{k+1}(t) -V_k(t)$')
        ax[0][1].set_xlabel('episode')

        # direct pathway
        for a in range(n_actions):
            ax[1][0].plot(avgA[0,a,:], color = colors[a], label = labels[a])
            ax[1][0].fill_between(x1, avgA[0,a,:] - varA[0,a,:], avgA[0,a,:] + varA[0,a,:], color = colors[a], alpha = 0.1)
            
        ax[1][0].set_title('Direct pathway')    
        ax[1][0].set_ylabel('$A(s_t,a)$')
        ax[1][0].set_xlabel('$s_t$')
        
        # indirect pathway
        for a in range(n_actions):
            ax[1][1].plot(avgA[1,a,:], color = colors[a], label = labels[a])
            ax[1][1].fill_between(x1, avgA[1,a,:] - varA[1,a,:], avgA[1,a,:] + varA[1,a,:], color = colors[a], alpha = 0.1)
        ax[1][1].set_title('Indirect pathway')    
        ax[1][1].set_ylabel('$A(s_t,a)$')
        ax[1][1].set_xlabel('$s_t$')
        
        # convergence - direct pathway
        for a in range(3):
            ax[2][0].plot(x2, avgDiffA[0,a,:], color = colors[a], label = labels[a])
            ax[2][0].fill_between(x2, avgDiffA[0,a,:] - varDiffA[0,a,:], avgDiffA[0,a,:] + varDiffA[0,a,:], color = colors[a], alpha = 0.1)
        ax[2][0].set_title('Direct pathway convergence')    
        ax[2][0].set_ylabel('$A_{k+1}(t) -A_k(t)$')
        ax[2][0].set_xlabel('episode')
        
        for a in range(3):
            ax[2][1].plot(x2, avgDiffA[1,a,:], color = colors[a], label = labels[a])
            ax[2][1].fill_between(x2, avgDiffA[1,a,:] - varDiffA[1,a,:], avgDiffA[1,a,:] + varDiffA[1,a,:], color = colors[a], alpha = 0.1)
        ax[2][1].set_title('Indirect pathway convergence')    
        ax[2][1].set_ylabel('$A_{k+1}(t) -A_k(t)$')
        ax[2][1].set_xlabel('episode')

        plt.legend()
        plt.show()

        fig.savefig("transfer_pathways_gridplot.png", bbox_inches='tight', dpi = 400)