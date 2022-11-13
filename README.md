# ML-WaferFault-Detection
###### Problem Statement: 
To build a classification methodology to predict the quality of wafer sensors based on the given training data. 

###### Description:
Manufacturer has deployed various wafers that are used in a semiconductor industry. Currently, if a wafer is faulty the entire manufacturing plant has to be stopped and each wafer has to be checked. In order to avoid shutting down all the lanes, sensors are deployed for each wafer. Sensors send signals that will help to identify the faulty wafer. A faulty wafer is labelled as -1 and intact one as +1. A machine learning algorithm needs to deployed to find the faulty wafer based on sensor signals.


# Architecture
courtesy: iNeuron Intelligence
![img](https://github.com/nds-najam/ML-WaferFault-Detection/blob/main/architecture.png)

Steps:
- Data Ingestion:
    - data for batch training
    - data validation
    - data transformation
    - data insertion in db
- data processing
    - export data from db to csv for training
    - data preprocessing
    - data clustering
- Model Training
    - best model for each cluster
    - hyperparameter tuning
    - model saving
- Deployment
    - Cloud setup
    - Pushing app to cloud
    - Application start
- Prediction
    - data ingestion
    - data processing
    - model call for specific cluster
    - prediction
    - export prediction to csv

## Workflow
1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration Manager in src config
6. Update the components
7. Update the pipeline
8. Test run pipeline stage
9. run tox for testing your package
10. run "dvc repro" for running all the stages in pipeline