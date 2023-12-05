
import json
import os
import math
import copy
from Dynamics.Dynamic import *
from Dynamics.trajectory import create_trajectory, initialize_game


def simulated_annealing(params, objective_function, initial_temperature,n_cooling,cooling_rate, iter_tolerance,
                        max_iterations_local):
    
    experimental_matrix = load_experimental_matrix()

    iter_local = 0
    iter_max_local = 0
    i = 0

    linked_params = ["ASPIRATION","PNB_VALUE"]

    current_params = params.copy()

    current_energy = objective_function(current_params,experimental_matrix)

    best_params = current_params.copy()
    best_energy = current_energy

    temperature = initial_temperature
    while iter_max_local < iter_tolerance:
        new_params = [generate_neighbor(params) for params in current_params]
        for lp in linked_params:
            new_params[0]["AG_VARS"][lp] = new_params[1]["AG_VARS"][lp]

        if iter_local >= max_iterations_local:
            new_params = [generate_random_game_dict(prior=params["AG_VARS"]) for params in best_params]
            for lp in linked_params:
                new_params[0]["AG_VARS"][lp] = new_params[1]["AG_VARS"][lp]
            #new_params["AG_VARS"]["BETA"] = np.zeros(new_params["STATIC_CONSTANTS"]["N_AGENTS"])
            iter_local = 0
            iter_max_local += 1
            if iter_max_local % (iter_tolerance//10) == 0:
                print(f"{iter_max_local/iter_tolerance*100} % THRESHOLD REACHED")

            #print("calling generate_random_game_dict ...")

        new_energy = objective_function(new_params,experimental_matrix)
        delta_energy = new_energy - current_energy
        if delta_energy < 0:
            current_params = new_params
            current_energy = new_energy


            if new_energy < best_energy:
                best_params = new_params
                best_energy = new_energy


                print(f"ITERATION: {i}, BEST_ENERGY: {best_energy:.2e}, TEMPERATURE:{temperature:.2f},ITER_LOCAL:{iter_local}.ITER_MAX_LOCAL:{iter_max_local}")
                print("AUTH PARAMS")
                print(best_params[0]["AG_VARS"])
                print("NON_AUTH PARAMS")
                print(best_params[1]["AG_VARS"])
                print("\n")

                iter_local = 0
                iter_max_local = 0
        else:

            probability = math.exp(-delta_energy / temperature)

            if rd.random() < probability:
                current_params = new_params
                current_energy = new_energy

        if i%n_cooling ==0:
            temperature *= cooling_rate
        i+=1
        iter_local += 1


    return best_params,best_energy



def generate_neighbor(params):
    # #Genera un vecino aleatorio cambiando los parámetros de uno de los individuos
    neighbor_params = copy.deepcopy(params)
    loop_keys = [key for key in params["AG_VARS"].keys() if key not in ["BETA","GLOB_VALUE"]]
    param_pos = rd.randint(0,params["STATIC_CONSTANTS"]["N_AGENTS"]-1)
    param_name = rd.choice(loop_keys)
    range_dict = {"ALPHA":(0,1,0.1),"BETA":(0,1,0.1),"ASPIRATION":(0,300,30),"PNB_VALUE":(0,30,5)}
    #for param_name in loop_keys:

    v_p = min(neighbor_params["AG_VARS"][param_name][param_pos] + range_dict[param_name][2],range_dict[param_name][1])
    v_n = max(neighbor_params["AG_VARS"][param_name][param_pos] - range_dict[param_name][2],range_dict[param_name][0])
    param_value = np.random.choice([v_p,v_n])
    #param_value = np.random.choice(np.arange(*range_dict[param_name]))
    neighbor_params["AG_VARS"][param_name][param_pos] = param_value
    if neighbor_params["AG_VARS"]["BETA"][param_pos] + neighbor_params["AG_VARS"]["ALPHA"][param_pos] > 1.01:
         if param_name == "ALPHA":
                p1,p2 = "ALPHA","BETA"
         elif param_name == "BETA":
                p1,p2 = "BETA","ALPHA"
         else:
            print(neighbor_params["AG_VARS"],param_name,param_pos)
         neighbor_params["AG_VARS"][p2][param_pos] = 1 - neighbor_params["AG_VARS"][p1][param_pos]


    return neighbor_params

def simulate_matrix(GAME_DICT):
    #start= time.perf_counter()
    total_tray = []
    for s in range(GAME_DICT["STATS_DICT"]["N_SIMS"]):
        #GAME_DICT = preprocess_game_dict(GAME_DICT)
        agent_matrix, GAME_DICT = initialize_game(GAME_DICT)
        tray = create_trajectory(agent_matrix, GAME_DICT, proc="for")
        total_tray.append(tray)
    final_tray = np.concatenate(total_tray, axis=2)
    lorenz = lorenz_map(final_tray, GAME_DICT)
    #print(f"SECONDS: {time.perf_counter()-start:.5f}")
    return lorenz

def objective_function(parameters, experimental_matrix):
    diff = 0
    for params,m in zip(parameters,experimental_matrix):
        simulated_matrix = simulate_matrix(params)  # Función que genera la matriz simulada
        # Función de pérdida que calcula la diferencia entre las matrices simulada y experimental
        diff += np.sum((simulated_matrix - m)**2)
    diff /= len(experimental_matrix)
    return diff


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def save_params(parameters,category,energy,file_name):
    params_to_save = dict()
    params_to_save[category] = dict()
    params_to_save[category]["params"] = {k: v for k, v in parameters["AG_VARS"].items()}
    params_to_save[category]["energy"] = energy

    if os.path.exists(file_name):
        with open(file_name) as f_load:
            file = json.load(f_load)
        if category in file.keys():
            if energy < file[category]["energy"]:
                save_flag = 1
            else:
                save_flag = 0
        else:
            save_flag = 1
        if save_flag:
            with open(file_name,"w") as f:
                file[category] = params_to_save[category]
                json.dump(file,f,cls=NpEncoder)
    else:
        with open(file_name,"w") as f:
            json.dump(params_to_save,f,cls=NpEncoder)

def load_best_params(file_name):
    """Load the best set of params in the .json file"""
    with open(file_name,"r") as f:

        file = json.load(f)
        best_energy = 10
        for name,item in file.items():
            energy = item["energy"]
            params = item["params"]
            if energy <= best_energy:
                best_energy = energy
                if "joined_auth" in name:
                    auth_best_params = params
                if "joined_non_auth" in name:
                    non_auth_best_params = params
    return auth_best_params,non_auth_best_params


#file = data.json
def plot_best_parameters(file,formatted_category):
    best_params = load_best_params(file)

    for i,params_ in zip(["auth","non_auth"],best_params):


        total_tray = []
        for s in range(30):
            params = generate_random_game_dict()
            params["AG_VARS"] = params_
            params["AG_VARS"]["ALPHA"] = np.array(params["AG_VARS"]["ALPHA"])
            params["AG_VARS"]["BETA"] = np.array(params["AG_VARS"]["BETA"])
            params["AG_VARS"]["ASPIRATION"] = np.array(params["AG_VARS"]["ASPIRATION"])
            params["AG_VARS"]["PNB_VALUE"] = np.array(params["AG_VARS"]["PNB_VALUE"])
            agent_matrix, GAME_DICT = initialize_game(params)
            tray = create_trajectory(agent_matrix, GAME_DICT, proc="for")
            total_tray.append(tray)
        final_tray = np.concatenate(total_tray, axis=2)

        plot_payoff_matrix(final_tray,GAME_DICT,"best_"+formatted_category)
        plot_lorenz_map(final_tray,GAME_DICT,"best_"+formatted_category)


def load_experimental_matrix():
    auth_experimental_lorenz = load_experimental_lorenz("experimental_data/df_experiment_global_with authority.csv")
    non_auth_experimental_lorenz = load_experimental_lorenz("experimental_data/df_experiment_global_without authority.csv")
    auth_experimental_payoff = load_experimental_payoff("experimental_data/df_experiment_global_with authority.csv")
    non_auth_experimental_payoff = load_experimental_payoff("experimental_data/df_experiment_global_without authority.csv")
    return [[auth_experimental_lorenz,non_auth_experimental_lorenz]]


def fit_params(save_file):
    category = "joined"
    linked_params = ["ASPIRATION","PNB_VALUE"]
    INITIAL_DICT = [generate_random_game_dict() for _ in range(2)]
    for lp in linked_params:
        INITIAL_DICT[0]["AG_VARS"][lp] = INITIAL_DICT[1]["AG_VARS"][lp]


    best_params,best_energy = simulated_annealing(INITIAL_DICT,objective_function,initial_temperature=1,n_cooling=1_000,cooling_rate=0.95\
                                    ,iter_tolerance=200,max_iterations_local=100)



    for i,params in zip(["auth","non_auth"],best_params):

        formatted_category = "_".join([category, i, "{:.3e}".format(best_energy)])
        save_params(params,formatted_category,best_energy,save_file)