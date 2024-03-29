'''
This script deploys a Prefect workflow named "LSTM best model deployment" 
using the set_workflow from the orchestrate_module module. 

'''

from orchestrate_module import set_workflow
from prefect.deployments import Deployment

deployment = Deployment.build_from_flow(
    flow= set_workflow,
    name= "LSTM best model deployment (current) v2.5"
)

if __name__== "__main__":
    deployment.apply()
