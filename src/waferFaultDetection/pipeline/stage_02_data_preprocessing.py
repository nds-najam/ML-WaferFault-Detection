from waferFaultDetection.config import ConfigurationManager
from waferFaultDetection.components import Preprocessor
from waferFaultDetection import logger

STAGE_NAME = "Data Preprocessing Stage"

def main():
    config = ConfigurationManager()
    data_preprocessing_config = config.get_data_preprocessing_config()
    data_preprocessing = Preprocessor(config=data_preprocessing_config)
    data_preprocessing.create_model_df()
    data_preprocessing.separate_label_features()
    data_preprocessing.check_missing_values()
    data_preprocessing.impute_missing_values()
    data_preprocessing.drop_columns_with_zero_std_dev()
    data_preprocessing.clusters_elbow_plot()
    data_preprocessing.create_clusters()
    data_preprocessing.get_preprocessed_model_input_file()

if __name__ == '__main__':
    try:

        logger.info(f"\n\n>>>>>>> stage: {STAGE_NAME} started <<<<<<<")
        main()
        logger.info(f">>>>>>> stage: {STAGE_NAME} completed <<<<<<< \n\n=====================")
    except Exception as e:
        logger.exception(e)
        raise e