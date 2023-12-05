from Dynamics.helper_functions import *
from Controller.plot_functions import *
import numpy as np
import random as rd

from Dynamics.trajectory import inhomogeneous_ranges


def generate_random_game_dict(prior = None):
    GAME_DICT = {}

    ### These are the constants that should be kept for all the game
    STATIC_CONSTANTS = {"N_AGENTS": 6,
                        "N_GROUPS":10,
                        "N_ACTIONS": 31,
                        "LEARNING_RATE": 1,
                        "MODE": "CPR"}

    GAME_DICT["STATIC_CONSTANTS"] = STATIC_CONSTANTS


    range_1_params = dict(population=list(range(0,150,30)), weights=None, cum_weights=None, k=STATIC_CONSTANTS["N_AGENTS"])
    range_2_params = dict(population=list(np.arange(0,1,0.1)), weights=None, cum_weights=None, k=STATIC_CONSTANTS["N_AGENTS"])
    range_3_params = dict(population=[5, 10, 15, 20, 25, 30], weights=None, cum_weights=None, k=STATIC_CONSTANTS["N_AGENTS"])
    range_4_params = dict(population=[14], weights=None, cum_weights=None, k=STATIC_CONSTANTS["N_AGENTS"])

    if prior != None:
        range_1_params["population"] = prior["ASPIRATION"]
        range_2_params["population"] = prior["ALPHA"]
        range_3_params["population"] = prior["PNB_VALUE"]
        range_4_params["population"] = prior["GLOB_VALUE"]

    ###These are the initial values for agents
    AG_VARS = {}
    GAME_DICT["AG_VARS"] = AG_VARS

    STATIC_AG_VARS = {
         "ASPIRATION_STATIC": {"func": rd.choices, "params": range_1_params},
         "ALPHA_STATIC": {"func": rd.choices, "params": range_2_params},
         "PNB_VALUE_STATIC": {"func": rd.choices, "params": range_3_params},
         #"GLOB_VALUE_STATIC":{"func": rd.choices, "params": range_4_params}
    }
    GAME_DICT["STATIC_AG_VARS"] = STATIC_AG_VARS

    ### These are the constants that should define the trajectory
    TRAJECTORY_DICT = {"N_STEPS": 70,
                       "TOLERANCE": 1.0,
                       "MEAN_SIZE": 35}

    GAME_DICT["TRAJECTORY_DICT"] = TRAJECTORY_DICT

    ### These are the constants that should define the statistics

    STATS_DICT = {"N_SIMS": 30}

    GAME_DICT["STATS_DICT"] = STATS_DICT


    #In order to introduce functions as arguments
    if "ASPIRATION_STATIC" in GAME_DICT.get("STATIC_AG_VARS",{}):
        GAME_DICT["AG_VARS"]["ASPIRATION"] =  inhomogeneous_ranges(STATIC_CONSTANTS, GAME_DICT["STATIC_AG_VARS"]["ASPIRATION_STATIC"])
    else:
        GAME_DICT["AG_VARS"]["ASPIRATION"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.full(STATIC_CONSTANTS["N_AGENTS"],30))

    if "ALPHA_STATIC" in GAME_DICT.get("STATIC_AG_VARS",{}):
        GAME_DICT["AG_VARS"]["ALPHA"] =  inhomogeneous_ranges(STATIC_CONSTANTS, GAME_DICT["STATIC_AG_VARS"]["ALPHA_STATIC"])
        GAME_DICT["AG_VARS"]["BETA"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.array([rd.choice([0.1*i for i in range(int(10*(1.1-value)))]) for value in GAME_DICT["AG_VARS"]["ALPHA"]]))

    else:
        GAME_DICT["AG_VARS"]["ALPHA"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.full(STATIC_CONSTANTS["N_AGENTS"],0.1))
        GAME_DICT["AG_VARS"]["BETA"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.full(STATIC_CONSTANTS["N_AGENTS"],0.5))

    #Introduce CPR values and possibility to introduce functional forms
    if GAME_DICT["STATIC_CONSTANTS"]["MODE"] == "CPR":

        if "PNB_VALUE_STATIC" in GAME_DICT.get("STATIC_AG_VARS",{}):
            GAME_DICT["AG_VARS"]["PNB_VALUE"] = inhomogeneous_ranges(STATIC_CONSTANTS,GAME_DICT["STATIC_AG_VARS"]["PNB_VALUE_STATIC"])
        else:
            GAME_DICT["AG_VARS"]["PNB_VALUE"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.full(STATIC_CONSTANTS["N_AGENTS"],15))

        if "GLOB_VALUE_STATIC" in GAME_DICT.get("STATIC_AG_VARS",{}):
            GAME_DICT["AG_VARS"]["GLOB_VALUE"] = inhomogeneous_ranges(STATIC_CONSTANTS,GAME_DICT["STATIC_AG_VARS"]["GLOB_VALUE_STATIC"])
        else:
            GAME_DICT["AG_VARS"]["GLOB_VALUE"] = inhomogeneous_ranges(STATIC_CONSTANTS,np.full(STATIC_CONSTANTS["N_AGENTS"],14))


    return GAME_DICT



def generate_ranges(DIC_RANGES,CONSTANTS_DICT):
    ranges_dict = {}
    for k,v in DIC_RANGES.items():

        if v[1] == "het":
            value = np.array([[0]*(CONSTANTS_DICT["N_AGENTS"] - item) + [0.5]*item \
             for item in v[0]])
            #value = np.array([lambda x=item: np.random.choice([0, 0.5],N_AGENTS, p=[x,1-x]) for item in v[0]])
            ranges_dict[k] = (value,"het","K_NORM")
        else:
            ranges_dict[k] = (v[0],"hom")

    return ranges_dict



#DIC_RANGES = generate_ranges(DIC_RANGES,CONSTANTS_DICT)



#print(parameter_exploration_2D(DIC_RANGES,CONSTANTS_DICT = CONSTANTS_DICT, mode="CPR",base_dir = "Report_CPR"))