from waferFaultDetection.constants import *
from waferFaultDetection.utils import read_yaml,create_directories
from waferFaultDetection import logger
from waferFaultDetection.entity import DataIngestionConfig
from waferFaultDetection.entity import DataPreprocessingConfig
from waferFaultDetection.entity import ModelTrainingConfig


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH
        ):

        logger.info("reading yaml files for configs and parameters")
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        logger.info("creating artifacts directory")
        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        logger.info("creating data ingestion root directory")
        create_directories([config.root_dir])

        logger.info('creating data ingestion configuration')
        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
            model_input_file= config.model_input_file
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing
        logger.info("creating data preprocessing root directory")
        create_directories([config.root_dir])

        logger.info('creating data preprocessing configuration')
        data_preprocessing_config = DataPreprocessingConfig(
            root_dir = config.root_dir,
            model_input_file = config.model_input_file,
            sensor_name_column = config.sensor_name_column,
            label_column_name = config.label_column_name,
            null_summary_file = config.null_summary_file,
            zero_stddev_columns_file = config.zero_stddev_columns_file,
            preprocessed_model_input_file = config.preprocessed_model_input_file,
            elbow_plot_file = config.elbow_plot_file,
            kmeans_model_file = config.kmeans_model_file
        )
        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        logger.info("creating model training root directory")
        create_directories([config.root_dir])

        logger.info('creating model training configuration')
        model_training_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            preprocessed_model_input_file = config.preprocessed_model_input_file,
            cluster_label = config.cluster_label,
            label_column_name = config.label_column_name,
            models_directory = config.models_directory
        )
        return model_training_config