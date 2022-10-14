import numpy as np
import pandas as pd
import pickle

def get_reward(state, action, m):
  '''
  Returns reward for a given tuple: (current state, action, next state)
  Inputs:
    state is the current state
    action is the action chosen
    m is the passive subsidy

  '''

  if state[0] == "L":
      reward = 1.0
  else:
      reward = -1
  if action == 'N':
      reward += m

  return reward

def convertAxis(T):
    '''
    convert T matrix from format: a, s, s' (where s=0 is bad state) 
                                    --> s, s', a (where s=0 is good state) 
    '''
    P=np.zeros_like(T)
    for a in range(2):
      for s in range(2):
        for ss in range(2):
          P[1-s,1-ss,a]=T[a,s,ss]
    return P
    


"""## Whittle Index functions"""

"""
Whittle index computation functions
"""


def planinf(two_state_probs, sleeping_constraint = True, GAMMA=0.99, modifiedV=False ):
    '''
    Implements sleeping constraint for inifinite months

    two_state_probs axes: action, starting_state, final_state. 
    State=1 means engaging state

    '''
    two_state_probs=convertAxis(two_state_probs)

    aug_states = []
    # L referes to Low-risk of dropout (E state)
    # H referes to High-risk of dropout (NE state)
    for i in range(2):
        if i % 2 == 0:
            aug_states.append('L{}'.format(i // 2))
        else:
            aug_states.append('H{}'.format(i // 2))

    if sleeping_constraint:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": aug_states + ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": 7,
            "gamma": GAMMA,
        }
    else:
        local_CONFIG = {
            'problem': {
                "orig_states": ['L', 'H'],
                "states": ['L', 'H'],
                "actions": ["N", "I"],
            },
            "time_step": 7,
            "gamma": GAMMA,
        }

    v_values = np.zeros(len(local_CONFIG['problem']['states']))
    q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                         len(local_CONFIG['problem']['actions'])))
    high_m_values = 2 * np.ones(len(local_CONFIG['problem']['states']))
    low_m_values = -2 * np.ones(len(local_CONFIG['problem']['states']))

    t_probs = np.zeros((len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['states']), \
                        len(local_CONFIG['problem']['actions'])))

    if sleeping_constraint:
        t_probs[0 : 2, 0 : 2, 0] = two_state_probs[:, :, 0] 
        t_probs[2 : 4, 2 : 4, 0] = two_state_probs[:, :, 0] 
        t_probs[0 : 2, 0 : 2, 1] = two_state_probs[:, :, 0] 
        t_probs[2 : 4, 0 : 2, 1] = two_state_probs[:, :, 1] 
    else:
        t_probs = two_state_probs

    max_q_diff = np.inf
    prev_m_values, m_values = None, None
    iteration=0
    while max_q_diff > 1e-5:
        #print ("Iteration: ",iteration)
        iteration=iteration+1
        prev_m_values = m_values
        m_values = (low_m_values + high_m_values) / 2
        #print ("m-values: ", m_values)
        if type(prev_m_values) != type(None) and \
        abs(prev_m_values - m_values).max() < 1e-20:
            break
        max_q_diff = 0
        v_values = np.zeros((len(local_CONFIG['problem']['states'])))
        q_values = np.zeros((len(local_CONFIG['problem']['states']), \
                             len(local_CONFIG['problem']['actions'])))
        delta = np.inf
        while delta > 0.0001:
            delta = 0
            for i in range(t_probs.shape[0]):
                v = v_values[i]
                v_a = np.zeros((t_probs.shape[2],))
                for k in range(v_a.shape[0]):
                    for j in range(t_probs.shape[1]):
                        v_a[k] += t_probs[i, j, k] * \
                        (get_reward(local_CONFIG['problem']['states'][i], \
                                    local_CONFIG['problem']['actions'][k], \
                                    m_values[i]) \
                         + local_CONFIG["gamma"] * v_values[j])
                        

                v_values[i] = np.max(v_a)
                delta = max([delta, abs(v_values[i] - v)])
        
        state_idx = -1
        for state in range(q_values.shape[0]):
            for action in range(q_values.shape[1]):
                for next_state in range(q_values.shape[0]):
                    q_values[state, action] += t_probs[state, next_state, action]\
                     * (get_reward(local_CONFIG['problem']['states'][state], \
                                   local_CONFIG['problem']['actions'][action], \
                                   m_values[state])\
                         + local_CONFIG["gamma"] * v_values[next_state])
            # print(state, q_values[cluster, state, 0], q_values[cluster, state, 1])

        for state in range(q_values.shape[0]):
            if abs(q_values[state, 1] - q_values[state, 0]) > max_q_diff:
                state_idx = state
                max_q_diff = abs(q_values[state, 1] - q_values[state, 0])

        #print("Q values: ", q_values)
        #print ("V value: ", v_values)
        # print(low_m_values, high_m_values)
        if max_q_diff > 1e-5 and q_values[state_idx, 0] < q_values[state_idx, 1]:
            low_m_values[state_idx] = m_values[state_idx]
        elif max_q_diff > 1e-5 and q_values[state_idx, 0] > q_values[state_idx, 1]:
            high_m_values[state_idx] = m_values[state_idx]

        # print(low_m_values, high_m_values, state_idx)
        # ipdb.set_trace()
    
    m_values = (low_m_values + high_m_values) / 2
    
    #return q_values, m_values
    return [m_values[-1], m_values[-2]] ## Whittle Indices Corresponding to NE and E state respectively

