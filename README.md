# Codebase for SAHELI

`data_description.md` contains the data schema of training data used by SAHELI.

`pipeline.py` performs all the steps of the SAHELI pipeline. From fetching data and computing whittle indices to ranking beneficiaires and pushing the scheduled list of beneficiaires to deliver service calls.

`whittle_utils.py` contains helper functions to compute whittle indices for solving the Restless Multi-Armed Bandit Problem.


Contact: For any queries, please contact the authors at vermashresth@google.com, gargisingh@google.com, lovishm@google.com, aditya_mate@g.harvard.edu
