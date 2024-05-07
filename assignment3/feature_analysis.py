import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination


# Load data
data = pd.read_csv('precode/SPRICE_Norwegian_Maritime_Data.csv')

# Select columns to include in the model (as determined by your analysis)
selected_columns = ['Air_temp_Act', 'Rel_Humidity_act', 'Rel_Air_Pressure', 'Wind_Speed_avg', 'Wind_Direction_vct']
data = data[selected_columns]

# Create a BIC Score object and perform Hill Climbing search
bic = BicScore(data)
hc = HillClimbSearch(data)
best_model = hc.estimate()

print("Best model structure: ", best_model.edges())


# Learning CPDs using Maximum Likelihood Estimators
model = BayesianNetwork(best_model.edges())
model.fit(data, estimator=MaximumLikelihoodEstimator)

for cpd in model.get_cpds():
    print("CPD for {}: ".format(cpd.variable))
    print(cpd)



# Perform inference
inference = VariableElimination(model)
# Query example: Probability of high wind given sunny weather
result = inference.query(variables=['Wind_Speed_avg'], evidence={'Weather': 'Sunny'})
print(result)
