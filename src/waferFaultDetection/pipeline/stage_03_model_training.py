from waferFaultDetection.config import ConfigurationManager
from waferFaultDetection.components import ModelTrainer
from waferFaultDetection import logger

STAGE_NAME = "Model Training Stage"

def main():
    config = ConfigurationManager()
    model_training_config = config.get_model_training_config()
    model_trainer = ModelTrainer(config=model_training_config)
    model_trainer.train_model()

if __name__ == '__main__':
    try:

        logger.info(f"\n\n>>>>>>> stage: {STAGE_NAME} started <<<<<<<")
        main()
        logger.info(f">>>>>>> stage: {STAGE_NAME} completed <<<<<<< \n\n=====================")
    except Exception as e:
        logger.exception(e)
        raise e