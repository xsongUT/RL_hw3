import numpy as np
import gymnasium as gym
from typing import Iterable, Tuple

from interfaces.policy import Policy

def off_policy_mc_prediction_weighted_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs: Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi: Policy,
    pi: Policy,
    initQ: np.ndarray,
    gamma: float = 1.0
) -> np.ndarray:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *weighted* importance sampling. 

    The algorithm can be found in Sutton & Barto 2nd edition p. 110.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]

    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The discount factor."""
    Q: np.ndarray = initQ
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA))
    """The importance sampling ratios."""

    ## TODO:
    # Implement the off-policy Monte-Carlo prediction algorithm using WEIGHTED importance sampling.
    # Hints:
    #   - Sutton & Barto 2nd edition p. 110
    #   -  Be sure to carefully follow the algorithm.
    #   -  Every-visit implementation is fine.
    #   -  Look at `reversed()` to iterate over a trajectory in reverse order.
    #   -  You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
    
    # Loop over each trajectory in the set of trajectories
    for traj in trajs:
        G = 0  # Return (reward) accumulator
        W = 1  # Importance sampling weight accumulator
        """ W represents the importance sampling weight, which adjusts for the discrepancy 
               between the behavior policy bpi (used to generate the data) and the target policy pi 
               (the one being evaluated).
            At the start of processing a trajectory, we haven't applied any corrections yet, so W
               starts as 1 (meaning no correction applied). 
            As we move backward through the trajectory, we adjust w at each step by multiplying it 
            by the ratio of action probabilities between the target policy and the behavior policy.
             W_t = W_(t+1) * π(a_t/S_t)/b(a_t/S_t)
            At the first step, this is:
             W = 1 * π(a_0/S_0)/b(a_0/S_0) = π(a_0/S_0)/b(a_0/S_0)

             This initialization allows us to accumulate the product of importance sampling 
             ratios as we step backward through the trajectory.
               """

        # Loop over trajectory in reverse order
        # Loop for each step of episode, t = T −1, T −2, . . . , 0, while W != 0:
        for (s_t, a_t, r_t1, s_t1) in reversed(traj):
            G = gamma * G + r_t1  # Update the return G

            # Update the cumulative sum of weights for state-action pair
            C[s_t, a_t] += W

            # Update Q with weighted importance sampling
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])

            # Calculate the importance sampling ratio
            W *= pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)

            # If W becomes 0, break out of the loop
            """
            w becomes 0 when the target policy pi assigns a zero probability to a certain action 
            that was taken by the behavior policy bpi. This means that the action taken in that 
            step is impossible under the target policy pi
            """
            if W == 0:
                break
    return Q

def off_policy_mc_prediction_ordinary_importance_sampling(
    observation_space: gym.spaces.Discrete,
    action_space: gym.spaces.Discrete,
    trajs:Iterable[Iterable[Tuple[int,int,int,int]]],
    bpi:Policy,
    pi:Policy,
    initQ:np.array,
    gamma:float = 1.0
) -> np.array:
    """
    Evaluate the estimated Q values of the target policy using off-policy Monte-Carlo prediction algorithm
    with *ordinary* importance sampling. 

    The algorithm with weighted importance sampling can be found in Sutton & Barto 2nd edition p. 110.
    You will need to make a small adjustment for ordinary importance sampling.

    Carefully look at page 109.

    Every-visit implementation is fine.

    Parameters:
        env_spec (EnvSpec): environment spec
        trajs (list): A list of N trajectories generated using behavior policy bpi
            - Each element is a tuple representing (s_t, a_t, r_{t+1}, s_{t+1})
        bpi (Policy): behavior policy used to generate trajectories
        pi (Policy): evaluation target policy
        initQ (np.ndarray): initial Q values; np array shape of [nS, nA]
        
    Returns:
        Q (np.ndarray): $q_pi$ function; numpy array shape of [nS, nA]
    """
    nS: int = observation_space.n
    """The number of states in the environment."""
    nA: int = action_space.n
    """The number of actions in the environment."""
    Q: np.ndarray = initQ
    """The Q(s, a) function to estimate."""
    C: np.ndarray = np.zeros((nS, nA))
    """The importance sampling ratios."""

    ## TODO:
    # Implement the off-policy Monte-Carlo prediction algorithm using ORDINARY importance sampling.
    # Hints:
    #   - Sutton & Barto 2nd edition p. 110 for the main algorithm.
    #   -  You will need to make a small adjustment for ordinary importance sampling. Carefully look at page 109.
    #        Consider how the C update might be different.
    #   -  Be sure to carefully follow the algorithm.
    #   -  Every-visit implementation is fine.
    #   -  Look at `reversed()` to iterate over a trajectory in reverse order.
    #   -  You can use the `pi.action_prob(state, action)` and `bpi.action_prob(state, action)` methods to get the action probabilities.
    # Loop over each trajectory in the set of trajectories
    for traj in trajs:
        G = 0  # Return (reward) accumulator
        W = 1  # Importance sampling weight accumulator
        # Loop over trajectory in reverse order
        # Loop for each step of episode, t = T −1, T −2, . . . , 0, while W != 0:
        for (s_t, a_t, r_t1, s_t1) in reversed(traj):
            G = gamma * G + r_t1  # Update the return G

            # Update the cumulative sum of weights for state-action pair
            #The denominater is the count of weight w for state-action pair
            C[s_t, a_t] += 1

            # Update Q with weighted importance sampling
            Q[s_t, a_t] += (W / C[s_t, a_t]) * (G - Q[s_t, a_t])

            # Calculate the importance sampling ratio
            W *= pi.action_prob(s_t, a_t) / bpi.action_prob(s_t, a_t)

            # If W becomes 0, break out of the loop
            """
            w becomes 0 when the target policy pi assigns a zero probability to a certain action 
            that was taken by the behavior policy bpi. This means that the action taken in that 
            step is impossible under the target policy pi
            """
            if W == 0:
                break





    return Q
