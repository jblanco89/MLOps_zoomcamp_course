from orchestrate import set_workflow
from prefect.deployments import Deployment

deployment = Deployment.build_from_flow(
    flow= set_workflow,
    name= "LSTM best model deployment (current) v2.1"
)

if __name__== "__main__":
    deployment.apply()