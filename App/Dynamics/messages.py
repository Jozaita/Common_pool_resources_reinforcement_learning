import numpy as np
from collections.abc import Iterable


def delta_negative_stimulus(actions,special_value):
    """
    Provides a negative stimuli (-1) on all the actions distinct to
    the chosen one.
    """
    if isinstance(special_value,int) or isinstance(special_value,float):
        return -(actions != np.full(len(actions),special_value)).astype(int)
    if isinstance(special_value,Iterable):

        return -(actions != np.array(special_value)).astype(int)

def delta_positive_stimulus(actions,special_value):
    """
    Provides a negative stimuli (-1) on all the actions distinct to
    the chosen one.
    """
    if isinstance(special_value,int) or isinstance(special_value,float):
        return (actions == np.full(len(actions),special_value)).astype(int)
    if isinstance(special_value,Iterable):

        return (actions == np.array(special_value)).astype(int)


def previous_round_stimulus(actions,actions_prev,special_value):
    """
    Provides a positive stimuli in an adjacent value of the previous one.
    If the previous value is above 14, the privileged va
    :param actions:
    :param actions_prev:
    :return:
    """

    if isinstance(special_value,int) or isinstance(special_value,float):
        preferred_value = np.array([x + np.sign(14 - x) for x in actions_prev])
        return -(actions != preferred_value).astype(int)

    else:
        return NotImplemented


def previous_round_accumulated_stimulus(actions,actions_prev,special_value):
    """
    Provides a positive stimuli if actions < actions_prev for values greater than special
    value and viceversa.
    :param actions:
    :param actions_prev:
    :return:
    """

    if isinstance(special_value,int) or isinstance(special_value,float):
        return np.logical_or(\
            np.logical_and(actions_prev>special_value,np.less(actions,actions_prev)), \
    np.logical_and(actions_prev<special_value,np.greater(actions,actions_prev))
        ).astype(int)
    else:
        return NotImplemented



def average_stimulus(actions,special_value):
    """
    Provides an stimuli as a result of the difference between the optimal contribution and the average contribution
    """

    if isinstance(special_value,int) or isinstance(special_value,float):
        arr = actions - special_value
        print(actions, arr/np.max(np.abs(arr)))
        return arr/np.max(np.abs(arr))
    else:
        return NotImplemented
