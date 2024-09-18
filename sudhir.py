import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Load the dataset
data = pd.read_csv("hypothyroid.data")  # Replace with your dataset path

# Create a Bayesian model based on domain knowledge or learned structure
model = BayesianModel([('T3', 'Hyperthyroid'), ('T4', 'Hyperthyroid'), ('TSH', 'Hyperthyroid'),
                       ('Goiter', 'Hyperthyroid'), ('T3', 'Hypothyroid'), ('T4', 'Hypothyroid'),
                       ('TSH', 'Hypothyroid'), ('Goiter', 'Hypothyroid')])

# Learn the parameters from the data
mle = MaximumLikelihoodEstimator(model, data)
model.fit(data, estimator=mle)

# Perform inference for a new instance
query = {'T3': 'high', 'T4': 'low', 'TSH': 'high', 'Goiter': 'yes'}
infer = VariableElimination(model)
result = infer.query(variables=['Hyperthyroid', 'Hypothyroid'], evidence=query)

print(result)