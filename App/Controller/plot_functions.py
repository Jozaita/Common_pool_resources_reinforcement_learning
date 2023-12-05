import numpy as np

import matplotlib.pyplot as plt

from Dynamics.helper_functions import *

import time
###TRAJECTORY PLOTS

def plot_payoff_matrix(result,n_frames,CONSTANTS_DICT):
    """
    :param result: Raw results from a trajectory
    :param n_frames: Number of frames
    :param CONSTANTS_DICT: Constants diccionary
    :return: Quiver resulting the movements in payoff matrix
    """
    matriz = np.sum(result[0,:,:].T,axis=1)
    tileada = np.tile(matriz, (result[0,:,:].T.shape[1], 1)).T
    scatter_values =  np.dstack((result[0,:,:].T,tileada))


    frames = np.linspace(0,scatter_values.shape[0]-1,n_frames,dtype=int)

    frames_to_save = scatter_values[frames,:,:]

    colors = cm.rainbow(np.linspace(0, 1, scatter_values.shape[1]))

    for i,matrix in enumerate(frames_to_save):
        fig,ax = plt.subplots()

        if i:
            diff_matrix = matrix - matrix_prev

            ax.quiver(matrix_prev[:,0],matrix_prev[:,1],
                       diff_matrix[:,0],diff_matrix[:,1],color=colors)


        plt.xlim(0,CONSTANTS_DICT["N_ACTIONS"])
        plt.ylim(0,CONSTANTS_DICT["N_AGENTS"]*CONSTANTS_DICT["N_ACTIONS"])
        plt.grid()
        fig.savefig(f"Report_CPR/Images_video/{i}.jpg")

        matrix_prev = matrix

####PARAMETER EXPLORATION


def parameter_exploration_2D(DIC_RANGES,*,CONSTANTS_DICT,mode="CPR",base_dir="Report_CPR"):

    PARAMETERS = []
    RANGES = []
    AUX_VARS = []

    for k,v in DIC_RANGES.items():
        if v[1] == "het":
            AUX_VARS.append(v[2])
        else:
            AUX_VARS.append("")
        PARAMETERS.append(k)
        RANGES.append(v[0])


    range1,range2 = RANGES
    param1,param2 = PARAMETERS


    agent_matrix,aspirations = initialize(CONSTANTS_DICT )
    decorator_trajectory = create_trajectory(CONSTANTS_DICT)
    decorator_measure = measure(CONSTANTS_DICT)
    trajectory = decorator_trajectory(one_round_dynamics)
    n_variables = decorator_measure(trajectory)(agent_matrix,aspirations,arg_dict=CONSTANTS_DICT,mode=mode).shape[0]
    results_matrix = np.zeros((len(range1),len(range2),n_variables))

    for i1,p1 in enumerate(range1):
        CONSTANTS_DICT[param1] = p1
        for i2,p2 in enumerate(range2):
            start = time.perf_counter()
            CONSTANTS_DICT[param2] = p2



            decorator_trajectory = create_trajectory(CONSTANTS_DICT)
            decorator_measure = measure(CONSTANTS_DICT)
            trajectory = decorator_trajectory(one_round_dynamics)

            agent_matrix,aspirations = initialize(CONSTANTS_DICT)
            stats = decorator_measure(trajectory)(agent_matrix,aspirations,arg_dict=CONSTANTS_DICT,mode=mode)
            results_matrix[i1,i2,:] = np.mean(stats,axis=1)

            param1_print,param2_print = param1,param2
            p1_print,p2_print = p1,p2
            if isinstance(p1,np.ndarray):
                p1_print = np.mean(p1)/0.5
                param1_print = AUX_VARS[0]
            if isinstance(p2,np.ndarray):
                p2_print = np.mean(p2)/0.5
                param2_print = AUX_VARS[1]


            print(f"{param1_print} = {p1_print :.2f}, {param2_print} = {p2_print:.2f}, time elapsed: {time.perf_counter() - start :.2f} seconds")

    measures = ("MEAN","STD")
    if mode == "CPR":
        variables = ("ACTION","PAYOFF")
        plot_titles = [var+"_"+mes for var in variables for mes in measures]
    else:
        raise Exception("Programmer need to include plot_titles")


    fig, ax = plt.subplots(nrows=n_variables//2, ncols=2, figsize=(20, 8))

    for j, m in enumerate(np.rollaxis(results_matrix,2)):
        i = (j // 2, j % 2)
        im = ax[i].imshow(m.T)
        plt.colorbar(im, ax=ax[i])
        if isinstance(range1[0],np.ndarray):
            ax[i].xaxis.set_ticks(range(0,len(range1),max(1,len(range1)//10)))
            ax[i].xaxis.set_ticklabels(["{:.1f}".format(np.mean(item)/0.5) for item in range1[::max(1,len(range1)//10)]],
                                   rotation=45)
            ax[i].set(xlabel=AUX_VARS[0])

        else:
            ax[i].xaxis.set_ticks(range(0,len(range1),max(1,len(range1)//10)))
            ax[i].xaxis.set_ticklabels(["{:1d}".format(int(item)) for item in range1[::max(1,len(range1)//10)]],
                                   rotation=45)
            ax[i].set(xlabel=param1)

        if isinstance(range2[0], np.ndarray):
            ax[i].yaxis.set_ticks(range(0,len(range2),max(1,len(range2)//10)))
            ax[i].yaxis.set_ticklabels(["{:.1f}".format(np.mean(item)/0.5) for item in range2[::max(1, len(range2) // 10)]],
                                       rotation=45)
            ax[i].set(ylabel=param2)
        else:
            ax[i].yaxis.set_ticks(range(0, len(range2), max(1, len(range2) // 10)))
            ax[i].yaxis.set_ticklabels(["{:.1f}".format(item) for item in range2[::max(1, len(range2) // 10)]],
                                       rotation=45)
            ax[i].set(ylabel=AUX_VARS[1])


        ax[i].set_title(plot_titles[j])


    plt.tight_layout()

    plt.savefig(f"{base_dir}/Heatmap_test_2_{CONSTANTS_DICT['LEARNING_RATE']}.jpg", bbox_inches="tight")

    return results_matrix

import matplotlib.pyplot as plt

def lorenz_map(data,GAME_DICT):
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    actions = data[0,:,:]
    all_tuples = []
    heat_matrix = np.zeros((STATIC_CONSTANTS["N_ACTIONS"],STATIC_CONSTANTS["N_ACTIONS"]))
    for arr in actions:
        tuples = [(arr[i],arr[i+1]) for i in range(arr.shape[0] - 1)]
        all_tuples.extend(tuples)

    for it in all_tuples:
        heat_matrix[int(it[0]),int(it[1])] += 1

    return heat_matrix / heat_matrix.sum()


def plot_lorenz_map(data,GAME_DICT,plot_title):
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    actions = data[0,:,:]
    all_tuples = []
    heat_matrix = np.zeros((STATIC_CONSTANTS["N_ACTIONS"],STATIC_CONSTANTS["N_ACTIONS"]))
    for arr in actions:
        tuples = [(arr[i],arr[i+1]) for i in range(arr.shape[0] - 1)]
        all_tuples.extend(tuples)

    for it in all_tuples:
        heat_matrix[int(it[0]),int(it[1])] += 1



    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlabel = "$a_{i}$",ylabel="$a_{i+1}$",
           xlim=(0, STATIC_CONSTANTS["N_ACTIONS"]),
           ylim=(0, STATIC_CONSTANTS["N_ACTIONS"]))
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    plt.imshow(heat_matrix)
    plt.colorbar()
    #plt.scatter(x,y)
    #plt.tight_layout()
    fig.savefig(f"images/Lorenz_map_{plot_title}.jpg",bbox_inches="tight")

def check_modulus(i,args):
    for arg in args:
        if i == arg:
            return True
    return False


def payoff_matrix(data,GAME_DICT):
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    actions = data[0, :, :]

    heat_matrix = np.zeros(
        (STATIC_CONSTANTS["N_ACTIONS"] * (STATIC_CONSTANTS["N_AGENTS"] - 1), STATIC_CONSTANTS["N_ACTIONS"]))


    all_tuples = np.array([(sum(act_round) - act_ag, act_ag) for act_round in actions.transpose() for i, act_ag in
                               enumerate(act_round)])

    for it in all_tuples:
        heat_matrix[int(it[0]), int(it[1])] += 1

    return heat_matrix /heat_matrix.sum()



def plot_payoff_matrix(data,GAME_DICT,name,split_asp=[]):
    STATIC_CONSTANTS = GAME_DICT["STATIC_CONSTANTS"]
    actions = data[0,:,:]

    heat_matrix = np.zeros((STATIC_CONSTANTS["N_ACTIONS"]*(STATIC_CONSTANTS["N_AGENTS"] - 1),STATIC_CONSTANTS["N_ACTIONS"]))

    if len(split_asp) > 0:

        all_tuples = np.array([(sum(act_round)-act_ag,act_ag) for act_round in actions.transpose() for i,act_ag in enumerate(act_round)\
            if check_modulus(i,split_asp)])
    else:
        all_tuples = np.array([(sum(act_round)-act_ag,act_ag) for act_round in actions.transpose() for i,act_ag in enumerate(act_round)])


    for it in all_tuples:
        heat_matrix[int(it[0]),int(it[1])] += 1



    fig = plt.figure(figsize=(10,10))
    ax = fig.add_axes([0,0,1,1])
    ax.set(xlabel = "Individual contribution to C.A.E.",ylabel="Group contribution to C:A.E.",
           xlim=(0, STATIC_CONSTANTS["N_ACTIONS"]),
           ylim=((STATIC_CONSTANTS["N_AGENTS"]-1)*STATIC_CONSTANTS["N_ACTIONS"],0))
    ax.xaxis.get_label().set_fontsize(20)
    ax.yaxis.get_label().set_fontsize(20)
    plt.imshow(heat_matrix,aspect="auto")
    ax.grid(color='w', linestyle='-', linewidth=2)

    #plt.colorbar()

    fig.savefig(f"images/{name}.png",bbox_inches="tight")
