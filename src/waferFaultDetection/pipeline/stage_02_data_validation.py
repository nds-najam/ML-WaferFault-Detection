from waferFaultDetection.config import ConfigurationManager
from waferFaultDetection.components import DataValidation
from waferFaultDetection import logger

STAGE_NAME = "Data Validation Stage"

def main():
    config = ConfigurationManager()
    data_validation_config = config.get_data_ingestion_config() 
    data_validation = DataValidation(config=data_validation_config)
    data_validation._validate_file_names()
    data_validation._validate_columns()

if __name__ == '__main__':
    try:

        logger.info(f"\n\n>>>>>>> stage {STAGE_NAME} started <<<<<<<")
        main()
        logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<< \n\n=====================")
    except Exception as e:
        logger.exception(e)
        raise e