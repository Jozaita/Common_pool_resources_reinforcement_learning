
"""Contains the functions for executing one round of the CPR game dynamics"""

import numpy as np
from collections.abc import Iterable
from Dynamics.messages import *

def decide(agent_matrix):
    """
    Actions are chosen for all the agents
    :param agent_matrix: Probability rates for each agent
    :return: Contributions from all agents
    """
    N = agent_matrix.shape[0]
    M = agent_matrix.shape[1]
    #Given matrix throw a list of random numbers
    rd_numbers = np.random.uniform(0,1-1e-4,size=N)
    #Select actions according to those numbers
    actions = [np.searchsorted(np.cumsum(row),rd_numbers[i]) for i,row in enumerate(agent_matrix)]
    #Return those actions
    return np.array(actions)

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
    payoffs = np.array([0 if disaster else N_ACTIONS - action - 1 for action in actions]) #-1 for normalization
    return payoffs

def feedback_climate(aspirations,payoffs,actions,CONSTANTS_DICT):

    stimula_econ =  (payoffs - aspirations) / np.maximum(aspirations,np.abs(CONSTANTS_DICT["N_ACTIONS"])\
                                                         - aspirations)
    stimula_norm = delta_negative_stimulus(actions,CONSTANS_DICT["N_ACTIONS"]//2)


    stimula = (1-CONSTANTS_DICT["ALPHA"])*stimula_econ + CONSTANTS_DICT["ALPHA"]*stimula_norm

    return stimula

####### Common Pool Resources Game
def collect_payoff_CPR(actions,GAME_DICT):
    """
    Generates payoff for population
    :param actions: actions taken by     individuals
    :param disaster: True if the disaster happens
    :return: Array of payoffs for individuals
    """
    #CPR case
    N_ACTIONS = GAME_DICT["STATIC_CONSTANTS"]["N_ACTIONS"]

    cae = 15*sum(actions) - 75/900*sum(actions)**2
    payoffs = np.array([N_ACTIONS - 1 - action + action/sum(actions)*cae if sum(actions)>0 \
        else 0 for action in actions])
    return payoffs


def feedback_CPR(payoffs,actions,GAME_DICT,actions_prev=None):
    aspirations = GAME_DICT["AG_VARS"]["ASPIRATION"]
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    alpha = GAME_DICT["AG_VARS"]["ALPHA"]
    beta = GAME_DICT["AG_VARS"]["BETA"]
    stimula_econ =  (payoffs - aspirations) / np.maximum(aspirations,15*STATIC_CONSTANTS["N_ACTIONS"]\
                                                         -75/900*STATIC_CONSTANTS["N_ACTIONS"]**2 - aspirations)


    stimula_norm = delta_positive_stimulus(actions,GAME_DICT["AG_VARS"]["PNB_VALUE"])


    if actions_prev is not None:
        stimula_glob = average_stimulus(actions, GAME_DICT["AG_VARS"]["GLOB_VALUE"][0])
    else:
        stimula_glob = average_stimulus(actions,GAME_DICT["AG_VARS"]["GLOB_VALUE"][0])


    if isinstance(alpha,float) and isinstance(beta,float):
        coef_e = 1 - alpha - beta
    elif isinstance(alpha,Iterable) and not(isinstance(beta,Iterable)):
        coef_e = np.ones(STATIC_CONSTANTS["N_AGENTS"]) - alpha - np.full(beta,STATIC_CONSTANTS["N_AGENTS"])
    elif isinstance(beta,Iterable) and not(isinstance(alpha,Iterable)):
        coef_e = np.ones(STATIC_CONSTANTS["N_AGENTS"]) - np.full(alpha,STATIC_CONSTANTS["N_AGENTS"]) - beta
    elif isinstance(alpha,Iterable) and isinstance(beta,Iterable):
        coef_e = np.ones(STATIC_CONSTANTS["N_AGENTS"]) - alpha - beta
    else:
        return NotImplemented

    stimula = coef_e * stimula_econ + alpha * stimula_norm + beta * stimula_glob


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
def one_round_dynamics(agent_matrix,GAME_DICT,actions_prev=None):
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    actions = decide(agent_matrix)
    if STATIC_CONSTANTS["MODE"] == "CLIMATE":
        disaster = check_disaster(actions, arg_dict)
        payoffs = collect_payoff_climate(actions, disaster, STATIC_CONSTANTS["N_ACTIONS"])
        stimula = feedback_climate(aspirations, payoffs, actions, GAME_DICT)
        update_probability_vector(agent_matrix, actions, stimula, STATIC_CONSTANTS["LEARNING_RATE"])
        return agent_matrix,actions,payoffs,disaster,stimula
    elif STATIC_CONSTANTS["MODE"] == "CPR":
        payoffs = collect_payoff_CPR(actions, GAME_DICT)
        stimula = feedback_CPR(payoffs, actions, GAME_DICT,actions_prev)
        update_probability_vector(agent_matrix, actions, stimula, STATIC_CONSTANTS["LEARNING_RATE"])
        return agent_matrix, actions, payoffs, stimula

