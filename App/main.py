

from Controller.Fit import plot_best_parameters
from Dynamics.one_round_dynamics import *
from Dynamics.messages import *
from Dynamics.trajectory import *
from Dynamics.Dynamic import *

#######################
def main():
    plot_best_parameters("data.json","message_average")



if __name__ == "__main__":
    main()
#################################
# linked_params = ["ASPIRATION","PNB_VALUE"]
# INITIAL_DICT = [generate_random_game_dict() for _ in range(2)]
# for lp in linked_params:
#     INITIAL_DICT[0]["AG_VARS"][lp] = INITIAL_DICT[1]["AG_VARS"][lp]
# #INITIAL_DICT["AG_VARS"]["BETA"] = np.zeros(INITIAL_DICT["STATIC_CONSTANTS"]["N_AGENTS"])
#
# best_params,best_energy = simulated_annealing(INITIAL_DICT,objective_function,experimental_matrix,initial_temperature=1,n_cooling=1_000,cooling_rate=0.95\
#                                   ,iter_tolerance=200,max_iterations_local=100)
#
#
#
# for i,params in zip(["auth","non_auth"],best_params):
#
#     formatted_category = "_".join([category, i, "{:.3e}".format(best_energy)])
#     save_params(params,formatted_category,best_energy,"data_positive.json")
#
#
#
#     total_tray = []
#     for s in range(params["STATS_DICT"]["N_SIMS"]):
#         agent_matrix, GAME_DICT = initialize_game(params)
#         tray = create_trajectory(agent_matrix, GAME_DICT, proc="for")
#         total_tray.append(tray)
#     final_tray = np.concatenate(total_tray, axis=2)
#
#     plot_payoff_matrix(final_tray,GAME_DICT,"best_"+formatted_category)
#     plot_lorenz_map(final_tray,GAME_DICT,"best_"+formatted_category)
#################################

