from Dynamics.one_round_dynamics import *
from Dynamics.messages import *


from collections.abc import Iterable
import numpy as np


def create_agent_matrix(CONSTANTS_DICT):
    agent_matrix = np.full((CONSTANTS_DICT["N_AGENTS"],
                            CONSTANTS_DICT["N_ACTIONS"]), 1 / CONSTANTS_DICT["N_ACTIONS"])

    return agent_matrix

def inhomogeneous_ranges(CONSTANTS_DICT,aspirations):
    if isinstance(aspirations,float) or isinstance(aspirations,int):
        ag_aspirations = np.full(CONSTANTS_DICT["N_AGENTS"], float(aspirations))
    if isinstance(aspirations,Iterable) and not(isinstance(aspirations,dict)):
        ag_aspirations = np.fromiter(aspirations,float)
    if isinstance(aspirations,dict):
        ag_aspirations = np.array(aspirations["func"](**aspirations["params"]))

    return ag_aspirations


def initialize_game(GAME_DICT):

    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    agent_matrix = create_agent_matrix(STATIC_CONSTANTS)

    return agent_matrix,GAME_DICT

def create_trajectory(agent_matrix,GAME_DICT,proc="while"):

    ###
    TRAJECTORY_DICT = GAME_DICT["TRAJECTORY_DICT"]
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    MEAN_SIZE = TRAJECTORY_DICT["MEAN_SIZE"]
    TOLERANCE = TRAJECTORY_DICT["TOLERANCE"]
    N_STEPS = TRAJECTORY_DICT["N_STEPS"]

    ###
    if proc == "while":

        sumactions = np.zeros(MEAN_SIZE)
        _ = 0
        equil = -MEAN_SIZE
        results_temp = np.zeros((2,6,2))
        while True:
            agent_matrix, *variables, stimula = one_round_dynamics(agent_matrix,{**GAME_DICT["AG_VARS"],\
                                                                                 **GAME_DICT["STATIC_CONSTANTS"],
                                                                                 **GAME_DICT["CPR_CONSTANTS"]})
            #TODO: HERE WE ARE TALKING ABOUT CPR, BUT NEEDS TO BE IMPLEMENTED ACCORDING TO THE GAME
            # agent_matrix,actions,payoffs stimula

            actions = variables[0]
            sumactions[np.mod(_, MEAN_SIZE)] = sum(actions)
            if np.mod(_, MEAN_SIZE) == 0:
                equil_prev = equil
                equil = np.mean(sumactions)

                if np.abs(equil - equil_prev) < TOLERANCE:
                    return results_temp

                results_temp = np.zeros(
                    (len(variables), STATIC_CONSTANTS["N_AGENTS"], MEAN_SIZE))
                sumactions = np.zeros(MEAN_SIZE)

            for i, var in enumerate(variables):
                results_temp[i, :, np.mod(_, MEAN_SIZE)] = var
            _ += 1
            #TODO: If GAME_DICT changes from round to round, it needs to be implemented here

    elif proc == "for":

        LIMIT = N_STEPS - MEAN_SIZE
        if LIMIT < 0:
            raise Exception("Please, provide a mean size value greater than n_steps")

        results_temp = np.zeros(
            (2, STATIC_CONSTANTS["N_AGENTS"], MEAN_SIZE))

        agent_matrix, *variables, stimula = one_round_dynamics(agent_matrix, GAME_DICT)
        for _ in range(N_STEPS):
            agent_matrix, *variables, stimula = one_round_dynamics(agent_matrix, GAME_DICT,variables[0])
            if _ >= LIMIT:
                for i, var in enumerate(variables):
                    results_temp[i,:,np.mod(_,MEAN_SIZE)] = var

        return results_temp






