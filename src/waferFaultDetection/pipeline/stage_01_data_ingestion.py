from waferFaultDetection.config import ConfigurationManager
from waferFaultDetection.components import DataIngestion
from waferFaultDetection import logger

STAGE_NAME = "Data Ingestion Stage"

def main():
    config = ConfigurationManager()
    data_ingestion_config = config.get_data_ingestion_config()
    data_ingestion = DataIngestion(config=data_ingestion_config)
    data_ingestion.download_kaggle_dataset()
    data_ingestion.unzip_and_clean()
    data_ingestion.validate_file_names()
    data_ingestion.validate_columns()
    data_ingestion.create_model_input_file()

if __name__ == '__main__':
    try:

        logger.info(f"\n\n>>>>>>> stage: {STAGE_NAME} started <<<<<<<")
        main()
        logger.info(f">>>>>>> stage: {STAGE_NAME} completed <<<<<<< \n\n=====================")
    except Exception as e:
        logger.exception(e)
        raise e