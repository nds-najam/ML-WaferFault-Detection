from waferFaultDetection.constants import *
from waferFaultDetection.utils import read_yaml,create_directories
from waferFaultDetection import logger
from waferFaultDetection.entity import DataIngestionConfig


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