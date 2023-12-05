### Helper functions
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from Dynamics.messages import delta_negative_stimulus
from collections.abc import Iterable
import itertools



def grid_parameters(param_grid):
    keys_ = param_grid.keys()
    combinations = itertools.product(*param_grid.values())
    ds=[dict(zip(keys_,cc)) for cc in combinations]
    return ds

def load_experimental_payoff(root_dir):
    authority_dataset = pd.read_csv(root_dir)
    all_tuples = []
    authority_dataset["partial_contribution"] = authority_dataset["total_contribution"] -\
                                                authority_dataset["contribution"]
    for ronda in list(authority_dataset["round_number"])[:-1]:
        data_round = authority_dataset[authority_dataset["round_number"] == ronda]
        all_tuples.extend(data_round[["partial_contribution", "contribution"]].values)

    all_tuples = np.array(all_tuples)

    heat_matrix = np.zeros((31 * 5, 31))

    for tup in all_tuples:
        heat_matrix[tup[0], tup[1]] += 1

    return heat_matrix / heat_matrix.sum()

def load_experimental_lorenz(root_dir):
    authority_dataset = pd.read_csv(root_dir)
    all_tuples_2 = []

    for code in set(authority_dataset["code"]):
        data_participant = authority_dataset[authority_dataset["code"] == code]
        for i in range(35):
            curr_value = data_participant[data_participant["round_number"] == i]["contribution"].values
            next_value = data_participant[data_participant["round_number"] == i + 1]["contribution"].values
            if len(curr_value) and len(next_value):
                all_tuples_2.append([curr_value[0], next_value[0]])

    heat_matrix_2 = np.zeros((31, 31))

    for tup in all_tuples_2:
        heat_matrix_2[tup[0], tup[1]] += 1

    return heat_matrix_2 / heat_matrix_2.sum()

###

def decide(agent_matrix):
    """
    Actions are chosen for all the agents
    :param agent_matrix: Probability rates for each agent
    :return: Contributions from all agents
    """
    N = agent_matrix.shape[0]
    M = agent_matrix.shape[1]
    #Given matrix throw a list of random numbers
    rd_numbers = np.random.uniform(0,1,size=N)
    #Select actions according to those numbers
    actions = [np.searchsorted(np.cumsum(row),rd_numbers[i]) for i,row in enumerate(agent_matrix)]
    #Return those actions
    return actions

### Particular functions for the different configurations:

####### Climate change game
def check_disaster(actions,CONSTANTS_DICT):
    """
    Check if the disaster has been produced
    :param actions: Actions taken by individuals
    :param threshold: Contribution threshold for disaster
    :param disaster_probability: Probability of disaster if threshold not reached
    :return: True if disaster, false if not disaster
    """
    #Given actions, obtain global contribution
    contribution = sum(actions)
    #Apply threshold model for contribution
    return False if contribution >= CONSTANTS_DICT["THRESHOLD"] else np.random.uniform()<CONSTANTS_DICT["DISASTER_PROBABILITY"]

def collect_payoff_climate(actions,disaster,N_ACTIONS):
    """
    Generates payoff for population
    :param actions: actions taken by     individuals
    :param disaster: True if the disaster happens
    :return: Array of payoffs for individuals
    """
    payoffs = np.array([0 if disaster else N_ACTIONS - action for action in actions])
    return payoffs

def feedback_climate(aspirations,payoffs,actions,CONSTANTS_DICT):

    stimula_econ =  (payoffs - aspirations) / np.maximum(aspirations,np.abs(CONSTANTS_DICT["N_ACTIONS"])\
                                                         - aspirations)
    stimula_norm = delta_negative_stimulus(actions,CONSTANS_DICT["N_ACTIONS"]//2)


    stimula = (1-CONSTANTS_DICT["ALPHA"])*stimula_econ + CONSTANTS_DICT["ALPHA"]*stimula_norm

    return stimula

####### Common Pool Resources Game
def collect_payoff_CPR(actions,N_ACTIONS):
    """
    Generates payoff for population
    :param actions: actions taken by     individuals
    :param disaster: True if the disaster happens
    :return: Array of payoffs for individuals
    """
    #CPR case
    cae = 15*sum(actions) - 75/900*sum(actions)**2
    payoffs = np.array([N_ACTIONS - action + action/sum(actions)*cae if sum(actions)>0 \
           else 0 for action in actions])

    return payoffs


def feedback_CPR(aspirations,payoffs,actions,CONSTANTS_DICT):

    stimula_econ =  (payoffs - aspirations) / np.maximum(aspirations,15*CONSTANTS_DICT["N_ACTIONS"]\
                                                         -75/900*CONSTANTS_DICT["N_ACTIONS"]**2 - aspirations)

    stimula_norm = delta_negative_stimulus(actions,CONSTANTS_DICT["MAX_VALUE"])

    if isinstance(CONSTANTS_DICT["ALPHA"],float):
        stimula = (1-CONSTANTS_DICT["ALPHA"])*stimula_econ + CONSTANTS_DICT["ALPHA"]*stimula_norm
    if isinstance(CONSTANTS_DICT["ALPHA"],Iterable):
        stimula = (np.ones(CONSTANTS_DICT["N_AGENTS"]) - CONSTANTS_DICT["ALPHA"]) * stimula_econ \
                  + CONSTANTS_DICT["ALPHA"] * stimula_norm

    return stimula


def update_probability_vector(agent_matrix,actions,stimula,lr):
    """
    Updates probability vector for actions
    :param agent_matrix: Collection of probability vectors
    :param stimula: Collection of stimulus
    :return: None, just inplace updating probability vectors
    """
    reinforcement = ((np.heaviside(stimula,0) - agent_matrix[range(len(actions)),actions])*np.abs(stimula)*lr)


    #Dirac reinforcement
    agent_matrix[range(len(actions)),actions] += reinforcement
    #Normalization
    agent_matrix /= agent_matrix.sum(axis=1,keepdims=True)


###Scheme for dynamics:
### Define a one-round dynamics with the composition of X functions
def one_round_dynamics(agent_matrix,aspirations,*,arg_dict,mode):
    actions = decide(agent_matrix)
    if mode == "CLIMATE":
        disaster = check_disaster(actions, arg_dict)
        payoffs = collect_payoff_climate(actions, disaster, arg_dict["N_ACTIONS"])
        stimula = feedback_climate(aspirations, payoffs, actions, arg_dict)
        update_probability_vector(agent_matrix, actions, stimula, arg_dict["LEARNING_RATE"])
        return agent_matrix,actions,payoffs,disaster,stimula
    elif mode == "CPR":
        payoffs = collect_payoff_CPR(actions, arg_dict["N_ACTIONS"])
        stimula = feedback_CPR(aspirations, payoffs, actions, arg_dict)
        update_probability_vector(agent_matrix, actions, stimula, arg_dict["LEARNING_RATE"])
        return agent_matrix, actions, payoffs, stimula


def initialize(CONSTANTS_DICT):
    agent_matrix = np.full((CONSTANTS_DICT["N_AGENTS"], CONSTANTS_DICT["N_ACTIONS"]), 1 / CONSTANTS_DICT["N_ACTIONS"])

    if isinstance(CONSTANTS_DICT["ASPIRATION"],float):
        aspirations = np.full(CONSTANTS_DICT["N_AGENTS"], CONSTANTS_DICT["ASPIRATION"])
    if isinstance(CONSTANTS_DICT["ASPIRATION"],Iterable):
        aspirations = np.fromiter(CONSTANTS_DICT["ASPIRATION"],float)
    return agent_matrix,aspirations

### Test: trajectory function:


def dynamics_CPR(CONSTANTS):
    N_AGENTS = CONSTANTS[0]
    N_ACTIONS = CONSTANTS[1]
    LEARNING_RATE = CONSTANTS[2]
    ASPIRATION = CONSTANTS[3]
    N_SIMS = CONSTANTS[4]
    TOLERANCE = CONSTANTS[5]
    ALPHA = CONSTANTS[6]
    ###INITIALIZER
    agent_matrix = np.full((N_AGENTS,N_ACTIONS),1/N_ACTIONS)
    #agent_matrix = np.array([gauss(np.arange(N_ACTIONS),0,10) for _ in range(N_AGENTS)])
    aspirations = np.full(N_AGENTS,ASPIRATION)

    ###DYNAMICS
    #Comments on performance
    #
    mean_0 = np.zeros(N_ACTIONS)
    std_0 = np.zeros(N_ACTIONS)
    payoff_mean = 0
    payoff_std = 0
    actions_mean = 0
    actions_std = 0
    payoff_mean_hist = np.zeros(N_SIMS)
    payoff_std_hist = np.zeros(N_SIMS)
    actions_mean_hist = np.zeros(N_SIMS)
    actions_std_hist = np.zeros(N_SIMS)

    MEAN_SIZE = 1000
    sumactions = np.zeros(MEAN_SIZE)
    equil = -MEAN_SIZE
    max_value_list = [sum(collect_payoff_CPR(np.full(N_AGENTS, i), N_ACTIONS)) for i in range(N_ACTIONS)]
    MAX_VALUE = max_value_list.index(max(max_value_list))



    for tray in range(N_SIMS):
        _ = 0
        while True:
            _ += 1
            actions = decide(agent_matrix)
            payoffs = collect_payoff_CPR(actions,N_ACTIONS)
            stimula = feedback_CPR(aspirations, payoffs,actions,N_ACTIONS,ALPHA,MAX_VALUE)
            update_probability_vector(agent_matrix, actions, stimula,LEARNING_RATE)
            sumactions[np.mod(_,MEAN_SIZE)] = sum(actions)


            payoff_mean += np.average(payoffs)
            payoff_std += np.std(payoffs)
            actions_mean += np.average(actions)
            actions_std += np.std(actions)

            if np.mod(_, MEAN_SIZE) == 0 and _>0:
                equil_prev = equil
                equil = np.mean(sumactions)

                if np.abs(equil - equil_prev) < TOLERANCE:
                    break

                sumactions = np.zeros(MEAN_SIZE)


                payoff_mean = 0
                payoff_std = 0
                actions_mean = 0
                actions_std = 0


        payoff_mean /= MEAN_SIZE
        payoff_std /= MEAN_SIZE
        actions_mean /= MEAN_SIZE
        actions_std /= MEAN_SIZE

        payoff_mean_hist[tray] = payoff_mean
        payoff_std_hist[tray] = payoff_std
        actions_mean_hist[tray] = actions_mean
        actions_std_hist[tray] = actions_std

        mean_0 += np.average(agent_matrix,axis=0)
        std_0 += np.std(agent_matrix,axis=0)

    mean_0 /= N_SIMS
    std_0 /= N_SIMS


    RESULTS = mean_0,std_0,\
            payoff_mean_hist,\
            payoff_std_hist,\
            actions_mean_hist,\
            actions_std_hist
    return RESULTS


###Plot functions

def plot_hists(RESULTS,CONSTANTS,base_dir):
    N_ACTIONS = CONSTANTS[1]
    mean_0 = RESULTS[0]
    std_0 = RESULTS[1]
    disasters = RESULTS[2]
    payoff_mean = RESULTS[3]
    payoff_std = RESULTS[4]
    actions_mean = RESULTS[5]
    actions_std = RESULTS[6]


    fig,ax = plt.subplots(2,2,figsize=(16,12))
    ax[0,0].set_title("Mean Contribution")
    ax[0,0].plot(range(N_ACTIONS),mean_0,lw=3)
    ax[0,0].fill_between(range(N_ACTIONS),mean_0+std_0,mean_0-std_0,alpha=0.5)
    ax[0,0].set_ylim(bottom=0)
    ax[0,0].grid()
    ax[1,1].set_title("Disasters")
    ax[1,1].hist(disasters,range=(0,1),bins=50)
    ax[1,1].grid()
    ax[0,1].set_title("Payoffs")
    ax[0,1].hist(payoff_mean, range=(0, N_ACTIONS), bins=N_ACTIONS,label="Mean",alpha=0.5)
    ax[0,1].hist(payoff_std,range=(0,N_ACTIONS),bins=N_ACTIONS,label="Std",alpha=0.5)
    ax[0,1].legend()
    ax[0,1].grid()
    ax[1,0].set_title("Actions")
    ax[1,0].hist(actions_mean, range=(0, N_ACTIONS), bins=N_ACTIONS,label="Mean",alpha=0.5)
    ax[1,0].hist(actions_std,range=(0,N_ACTIONS),bins=N_ACTIONS,label="Std",alpha=0.5)
    ax[1,0].grid()
    ax[1,0].legend()

    plt.savefig("{0}/plot_hists_{1:.2f}_{2:.2f}".format(base_dir,CONSTANTS[5], CONSTANTS[3]) + ".jpg")

def create_meshgrid(ranges,n_points):
    """
    Returns the meshgrid for a parameter exploration
    :param ranges: lists with tuples of ranges
    :param n_points: Number of equally distributed points for this exploration
    :return: np.meshgrid of points
    """

    grid = np.meshgrid(*[np.linspace(x,y,n_points) for x,y in ranges])
    return np.dstack(grid)

def create_heatmap(var_names,grid,CONSTANTS,base_dir):
    """
    Grid is supposed to be 2D
    :param var_names: Name of the variables used for the heatmap
    :param grid: The grid from create_meshgrid
    :param CONSTANTS: Variables not changed
    :return: Heatmap plot
    """
    CONSTANTS_NAMES = ["N_AGENTS", "N_ACTIONS", "THRESHOLD", "DISASTER_PROBABILITY", "LEARNING_RATE", "ASPIRATION",
                       "N_SIMS",
                       "TOLERANCE", "ALPHA"]
    PLOT_NAMES = ["DISASTERS","MEAN_PAYOFF","STD_PAYOFF","MEAN_CONTRIBUTION","STD_CONTRIBUTION"]
    c_index = [CONSTANTS_NAMES.index(item) for item in var_names]

    disasters_matrix = np.zeros((grid.shape[0],grid.shape[1]))
    payoff_mean_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    payoff_std_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    actions_mean_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    actions_std_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    for i,xx in enumerate(grid):
        for j,yy in enumerate(xx):

            start = time.perf_counter()
            CONSTANTS[c_index[0]] = yy[0]
            CONSTANTS[c_index[1]] = yy[1]

            RESULTS = dynamics(CONSTANTS)

            disasters_matrix[i,j],payoff_mean_matrix[i,j],\
            payoff_std_matrix[i,j],actions_mean_matrix[i,j],\
            actions_std_matrix[i,j] = [np.average(item) for item in RESULTS[2:]]

            print(f"{time.perf_counter()-start:.2f} seconds for {var_names[0]} : {yy[0]:.1f}, {var_names[1]} : {yy[1]:.2f}")
    matrix = disasters_matrix,payoff_mean_matrix,payoff_std_matrix,\
        actions_mean_matrix,actions_std_matrix

    fig = plt.figure(figsize=(20,8))
    ax1 = plt.subplot2grid(shape=(2, 6), loc=(0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)
    ax4 = plt.subplot2grid((2, 6), (1, 1), colspan=2)
    ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=2)
    ax = [ax1,ax2,ax3,ax4,ax5]

    for i, m in enumerate(matrix):
        im = ax[i].imshow(m)
        plt.colorbar(im,ax=ax[i])

        ax[i].xaxis.set_ticks(range(grid.shape[0]))
        ax[i].xaxis.set_ticklabels(["{:d}".format(int(item)) for item in grid[0, :][:, 0]])
        ax[i].yaxis.set_ticks(range(grid.shape[1]))
        ax[i].yaxis.set_ticklabels(["{:.1f}".format(item) for item in grid[:,0][:, 1]])

        ax[i].set(xlabel=var_names[0], ylabel=var_names[1])

        ax[i].set_title(PLOT_NAMES[i])


    plt.savefig("{}/Heatmap_alpha{}_lr{}.jpg".format(base_dir,CONSTANTS[-1],CONSTANTS[-5]),bbox_inches="tight")


def create_heatmap_CPR(var_names,grid,CONSTANTS,base_dir):
    """
    Grid is supposed to be 2D
    :param var_names: Name of the variables used for the heatmap
    :param grid: The grid from create_meshgrid
    :param CONSTANTS: Variables not changed
    :return: Heatmap plot
    """
    CONSTANTS_NAMES = ["N_AGENTS", "N_ACTIONS", "LEARNING_RATE", "ASPIRATION",
                       "N_SIMS",
                       "TOLERANCE", "ALPHA"]
    PLOT_NAMES = ["MEAN_PAYOFF","STD_PAYOFF","MEAN_CONTRIBUTION","STD_CONTRIBUTION"]
    c_index = [CONSTANTS_NAMES.index(item) for item in var_names]


    payoff_mean_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    payoff_std_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    actions_mean_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    actions_std_matrix  = np.zeros((grid.shape[0],grid.shape[1]))
    for i,xx in enumerate(grid):
        for j,yy in enumerate(xx):

            start = time.perf_counter()
            CONSTANTS[c_index[0]] = yy[0]
            CONSTANTS[c_index[1]] = yy[1]

            RESULTS = dynamics_CPR(CONSTANTS)

            payoff_mean_matrix[i,j],\
            payoff_std_matrix[i,j],actions_mean_matrix[i,j],\
            actions_std_matrix[i,j] = [np.average(item) for item in RESULTS[2:]]

            print(f"{time.perf_counter()-start:.2f} seconds for {var_names[0]} : {yy[0]:.1f}, {var_names[1]} : {yy[1]:.2f}")
    matrix = payoff_mean_matrix,payoff_std_matrix,\
        actions_mean_matrix,actions_std_matrix

    fig,ax = plt.subplots(nrows=2,ncols=2,figsize=(20,8))


    for j, m in enumerate(matrix):
        i = (j//2,j%2)
        im = ax[i].imshow(m)
        plt.colorbar(im,ax=ax[i])

        ax[i].xaxis.set_ticks(range(0,grid.shape[0],grid.shape[0]//10))
        ax[i].xaxis.set_ticklabels(["{:d}".format(int(item)) for item in grid[0, :][:, 0][::grid.shape[0]//10]],rotation=45)
        ax[i].yaxis.set_ticks(range(0,grid.shape[1],grid.shape[1]//10))
        ax[i].yaxis.set_ticklabels(["{:.1f}".format(item) for item in grid[:,0][:, 1][::grid.shape[1]//10]],rotation=45)

        ax[i].set(xlabel=var_names[0], ylabel=var_names[1])

        ax[i].set_title(PLOT_NAMES[j])

    plt.tight_layout()

    plt.savefig("{}/Heatmap_zoomed_lr{}.jpg".format(base_dir,CONSTANTS[-1],CONSTANTS[-5]),bbox_inches="tight")












