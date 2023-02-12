import pandas as pd
import numpy as np
from waferFaultDetection import logger
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
import pickle
from waferFaultDetection.entity import ModelTrainingConfig
from sklearn.model_selection import train_test_split
import os

class ModelTrainer:
    """
        This class shall  be used to train the model and finding the best model for each cluster.

        Written By: Najam Sheikh
        Version: 1.0
        Revisions: None

        """
    def __init__(
        self,
        config:ModelTrainingConfig
    ):
        self.config = config
        self.rfc = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')

    def get_best_params_for_RFC(self,X_train,y_train):
        """
            Method Name: get_best_params_for_RFC
            Description: get the parameters for Random Forest Algorithm that gives the best accuracy.
                            Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception

            Written By: Najam
            Version: 1.0
            Revisions: None

        """
        logger.info("Entered the (get_best_params_for_RFC) module")
        try:
            # initializing the grid parameters
            logger.info('RFC:initializing the grid parameters')
            param_grid = {'n_estimators':[10,50,100,130],'criterion':['gini','entropy'],
                            'max_depth':range(2,4,1),'max_features':['sqrt','log2']}
            
            # Creating an object of the Grid Search Class and finding best parameters
            grid = GridSearchCV(estimator=self.rfc,param_grid=param_grid,cv=5,verbose=3)
            grid.fit(X_train,y_train)
            logger.info('RFC: Grid search and Fit complete')

            # Extracting best parameters
            n_estimators = grid.best_params_['n_estimators']
            criterion = grid.best_params_['criterion']
            max_depth = grid.best_params_['max_depth']
            max_features = grid.best_params_['max_features']
            logger.info('RFC:Extracting best parameters complete')

            # Creating a new model with the best parameters
            self.rfc = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,
                                                max_depth=max_depth,max_features=max_features)
            logger.info('RFC: Creating a new model with the best parameters complete')
            
            # Train the new model
            self.rfc.fit(X_train,y_train)
            logger.info('RFC: Training the new model complete')

            logger.info(f'Random Forest best parameters:{grid.best_params_}')
            logger.info('Exited the (get_best_params_for_RFC) module')
            return self.rfc

        except Exception as e:
            logger.info(f"Exception occured in (get_best_params_for_RFC) method. Exception message:{str(e)}")
            logger.info("Getting best parameters for Random Forest Classifier unsuccessful")
            raise e

    def get_best_params_for_XGB(self,X_train,y_train):
        """
            Method Name: get_best_params_for_XGB
            Description: get the parameters for XG Boost Algorithm that gives the best accuracy.
                            Use Hyper Parameter Tuning.
            Output: The model with the best parameters
            On Failure: Raise Exception

            Written By: Najam
            Version: 1.0
            Revisions: None

        """
        logger.info("Entered the (get_best_params_for_XGB) module")
        try:
            # initializing the grid parameters
            param_grid = {'learning_rate':[0.5, 0.1, 0.01, 0.001],
                            'max_depth':range(2,4,1),
                            'n_estimators': [10, 50, 100, 200]}
            
            # Creating an object of the Grid Search Class and finding best parameters
            grid = GridSearchCV(estimator=self.xgb,param_grid=param_grid,cv=5,verbose=3)
            grid.fit(X_train,y_train)
            logger.info('XGB Grid fit complete')

            # Extracting best parameters
            n_estimators = grid.best_params_['n_estimators']
            learning_rate = grid.best_params_['learning_rate']
            max_depth = grid.best_params_['max_depth']
            logger.info('XGB best parameters extraction complete')

            # Creating a new model with the best parameters
            logger.info('XGB Classifier: Creating a new model with the best parameters ')
            self.xgb = XGBClassifier(learning_rate=learning_rate,
                                    max_depth=max_depth,
                                    n_estimators=n_estimators)

            # Train the new model
            logger.info('XGB: Train the new model started')
            self.xgb.fit(X_train,y_train)
            logger.info(f'XG Boost Classifier best parameters:{grid.best_params_}')
            logger.info('Exited the (get_best_params_for_XGB) module')
            return self.xgb

        except Exception as e:
            logger.info(f"Exception occured in (get_best_params_for_XGB) method. Exception message:{str(e)}")
            logger.info("Getting best parameters for XG Boost Classifier unsuccessful")
            raise e

    def find_best_model(self,X_train,y_train,X_test,y_test):
        """
            Method Name: find_best_model
            Description: Find best model for each training and testing dataset with best AUC score
            Output: The model with the best parameters
            On Failure: Raise Exception

            Written By: Najam
            Version: 1.0
            Revisions: None

        """
        logger.info('Entered (find_best_model) method')
        try:
            # Create the best model for XGBoost and make prediction
            self.xgb = self.get_best_params_for_XGB(X_train,y_train)
            pred_xgb = self.xgb.predict(X_test)
            logger.info('XGB: Create the best model for XGBoost and make prediction')

            # if there's only one label in y, roc_auc_score fails. Instead use accuracy_score
            logger.info("if there's only one label in y, roc_auc_score fails. Instead use accuracy_score")
            if len(y_test.unique() == 1):
                xgb_score = accuracy_score(y_test,pred_xgb)
                logger.info(f'Accuracy score for XG Boost is {xgb_score}')
            else:
                xgb_score = roc_auc_score(y_test,pred_xgb) # AUC score
                logger.info(f'AUC score for XG Boost is {xgb_score}')
            logger.info('XGB: score calculation complete')

            # Create the best model for Random Forest Classifier and make prediction
            self.rfc = self.get_best_params_for_RFC(X_train,y_train)
            pred_rfc = self.rfc.predict(X_test)
            logger.info('RFC: Create the best model for Random Forest and make prediction')

            # if there's only one label in y, roc_auc_score fails. Instead use accuracy_score
            if len(y_test.unique() == 1):
                rfc_score = accuracy_score(y_test,pred_rfc)
                logger.info(f'Accuracy score for Random Forest Classifier is {rfc_score}')
            else:
                rfc_score = roc_auc_score(y_test,pred_rfc) # AUC score
                logger.info(f'AUC score for Random Forest Classifier is {rfc_score}')
            logger.info('RFC: score calculation complete')

            # Comparison of two models
            if(rfc_score < xgb_score):
                return 'XGBoost',self.xgb
            else:
                return 'RandomForest',self.rfc
            logger.info(f'XGBoost_Score:{xgb_score} \t RandomForest_Score:{rfc_score}')

        except Exception as e:
            logger.info(f"Exception occured in (find_best_model) method. Exception message:{str(e)}")
            logger.info("Finding of best model unsuccessful")
            raise e

    def train_model(self):
        """
                Method Name: train_model
                Description: This method trains individual cluster with best model and saves them.
                Output: A pickle model file for each cluster.
                On Failure: Raise Exception

                Written By: Najam Sheikh
                Version: 1.0
                Revisions: None

        """
        logger.info('Entered (create_model_df) method')
        try:
            data = pd.read_csv(self.config.preprocessed_model_input_file)
            list_of_clusters = data[self.config.cluster_label].unique()
            
            # replace the Good/bad labels to 0 and 1 from -1 and 1
            data[self.config.label_column_name].replace(to_replace={-1:0},inplace=True)
            logger.info('Replaced -1 with 0 for logistic regression')
            
            for cluster in list_of_clusters:
                # Create features matrix
                X = data[data[self.config.cluster_label] == cluster]
                X = X.drop(labels=[self.config.cluster_label,self.config.label_column_name],axis=1)
                logger.info('Created X features matrix')

                # Create labels vector
                y = data[data[self.config.cluster_label] == cluster]
                y = y[self.config.label_column_name]
                logger.info('Created y labels vector')

                # Split training and testing datasets
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=355)
                logger.info('Splitted X and y as training and testing datasets')

                # Find best model for the cluster
                best_model_name,best_model = self.find_best_model(X_train,y_train,X_test,y_test)

                # Saving best model to the directory
                if not os.path.exists(self.config.models_directory):
                    os.makedirs(self.config.models_directory)
                file = best_model_name + f'_cluster_{cluster}.sav'
                filepath = os.path.join(self.config.models_directory,file)
                with open(filepath,'wb') as f:
                    pickle.dump(best_model,f)

                logger.info(f'Saved best model:{best_model_name} for cluster:{cluster}')
                logger.info('Exited (train_model) method')
        except Exception as e:
            logger.info(f"Exception occured in (create_model_df) method. Exception message:{str(e)}")
            logger.info("Training of model unsuccessful")
            raise e