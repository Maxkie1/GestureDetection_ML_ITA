"""
The azure_utils.py script contains the Azure Machine Learning utilities.
"""

# Import the necessary libraries
from azure.ai.ml import MLClient
from azure.ai.ml import command
from azure.ai.ml.sweep import Choice, BayesianSamplingAlgorithm
from azure.identity import DefaultAzureCredential

# Print the docstring
print(__doc__)

# Enter the Azure Machine Learning workspace details
subscription_id = '9a7d9502-23ae-4685-b754-81039757f5e9'
resource_group = 'ML_ResourceGroup'
workspace_name = 'GestureDetection_ML_ITA'

# Create an Azure Machine Learning client
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace_name)

# Select compute target
compute_target = 'ML-ITA-ComputeCluster'
try:
    # Get the compute target
    compute_target = ml_client.compute_targets.get(compute_target)
    print('Compute target found: ', compute_target.name)
except:
    # Compute target not found
    print('Compute target not found')

# Define the job environment

# Define the hyperparameter space
hyperparameter_space = {
    'batch_size': Choice([32, 64, 128]),
    'epochs': Choice([10, 20, 30]),
    'first_layer_neurons': Choice([32, 64, 128]),
    'second_layer_neurons': Choice([32, 64, 128]),
    'third_layer_neurons': Choice([32, 64, 128]),
}








    
