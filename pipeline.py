'''
This script contains all steps performed in the SAHELI Piepline. 
0. Load the deployment parameters from a remote Table. This includes parameters such as the budget K and
registration time window of beneficiairies. (K is 1000 as per current constraints of the NGO)
1. Loading and preprocessing Beneficiary Data,
2. Loading a mapping model which predicts clusters given beneficiary features,
3. Loading pre-computed whittle index mapping from clusters
4. Predicting clusters
5. Fetching previous intervention Lists 
6. Obtaining current state for every beneficiary
7. Obtaining whittle index for every beneficiary based on cluster and current state
8. Ranking the Beneficiaries by Whittle Index
9. Pushing the list of top-K interventions
'''
import numpy as np
import pandas as pd
from datetime import datetime
import pandas as pd

from data_utils import load_data
from model_utils import load_mapping_model
from whittle_utils import load_precomputed_whittle_indices
from armman_db_utils import load_params_table, load_interventions_table, push_interventions

np.random.seed(1)

# Define augmented MDP
aug_states = []
for i in range(2):
    if i % 2 == 0:
        aug_states.append('L{}'.format(i // 2))
    else:
        aug_states.append('H{}'.format(i // 2))


df_params = load_params_table()

print('Define CONFIG Dictionary')
CONFIG = {
    "problem": {
        "orig_states": ['L', 'H'],
        "states": aug_states + ['L', 'H'],
        "actions": ["N", "I"],
    },
    "time_step": 7,
    "gamma": 0.99,
    "clusters": 20,
    "transitions": "weekly",
    "clustering": "kmeans",
    "start_date": datetime.today().strftime('%Y-%m-%d'),
    "data": "data",
    "interventions": int(df_params['beneficiary_count'].to_list()[-1]),
    "read_sql": 1, 
    "from_registration_date": str(df_params['start_date'].to_list()[-1]),
    "to_registration_date": str(df_params['end_date'].to_list()[-1]),
}

print('Load Data')
static_features, beneficiary_data, call_data , user_ids  = load_data(CONFIG)
print('Load Mapping Model')
cls, scaler = load_mapping_model(CONFIG)
print('Load Whittle Indices')
m_values = load_precomputed_whittle_indices(CONFIG)

print("Obtaining cluster predictions")
cluster_predictions = cls.predict(scaler.transform(static_features))

print("Obtaining intervention list")
df_intervention, intervention_users = load_interventions_table(CONFIG)

print("Obtaining States and Corresponding whittle indices")    
whittle_indices = {'user_id': [], 'whittle_index': [], 'cluster': [], 'start_state': [], 'registration_date': [], 'current_E2C': []}
for idx, puser_id in enumerate(user_ids):
    print('index: '+str(idx))
    date_num = (pd.to_datetime(CONFIG['start_date'], format="%Y-%m-%d") - pd.to_datetime("2018-01-01", format="%Y-%m-%d")).days
    
    if puser_id in intervention_users:
        continue
    
    # Obtain Calls in last 7 days
    past_days_calls = call_data[
    (call_data["user_id"]==puser_id)&
    (call_data["startdate"]<date_num)&
    (call_data["startdate"]>=date_num - 7)
]
    # Compute Connections and Engagement using Duration Threshold
    past_days_connections = past_days_calls[past_days_calls['duration']>0].shape[0]
    past_days_engagements = past_days_calls[past_days_calls['duration'] >= 30].shape[0]

    if past_days_engagements == 0:
        curr_state = 3
    else:
        curr_state = 2

    # Obtain Predicted Cluster for the beneficiary
    curr_cluster = cluster_predictions[idx]

    # Obtain Corresponding Whittle Index for the cluster and its current state
    whittle_indices['user_id'].append(puser_id)
    whittle_indices['whittle_index'].append(m_values[curr_cluster, curr_state])
    whittle_indices['cluster'].append(curr_cluster)
    if curr_state == 3:
        whittle_indices['start_state'].append('NE')
    elif curr_state == 2:
        whittle_indices['start_state'].append('E')
    
    regis_date = beneficiary_data[beneficiary_data['user_id'] == puser_id]['registration_date'].item()

    whittle_indices['registration_date'].append(regis_date)

print("finished computing whittle indices")
df = pd.DataFrame(whittle_indices)
df = df.sort_values('whittle_index', ascending=False)


print('Pushing New Intervention List')
df_int = df[:CONFIG["interventions"]]

push_interventions(df_int, CONFIG)
