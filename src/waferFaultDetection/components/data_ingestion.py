import os
import urllib.request as request
from zipfile import ZipFile
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from waferFaultDetection import logger
from waferFaultDetection.constants import *
from waferFaultDetection.entity import DataIngestionConfig
from waferFaultDetection.utils import get_size
from tqdm import tqdm

class DataIngestion:
    def __init__(
        self,
        config:DataIngestionConfig
    ):
        self.config = config

    def download_file(self):
        logger.info('Trying to download file')
        if not os.path.exists(self.config.local_data_file):
            try:
                logger.info("downloading the dataset ...")
                filename, headers = request.urlretrieve(
                    url = self.config.source_URL,
                    filename = self.config.local_data_file
                )
                logger.info('{filename} dataset download completed')
                logger.info('{filename} downloaded with following info: \n{headers}')
            except Exception as e:
                logger.info(f'{filename} download! dataset download failed')
                raise e
        else:
            logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")

    def download_kaggle_dataset(self):
        logger.info('Trying to download file')
        if not os.path.exists(self.config.local_data_file):
            try:
                logger.info("Authenticating Kaggle API")
                # Opening JSON file
                with open(KAGGLE_JSON_PATH) as json_file:
                    data = json.load(json_file)

                os.environ['KAGGLE_USERNAME']=data['username']
                os.environ['KAGGLE_KEY']=data['key']

                api = KaggleApi()
                api.authenticate()
                logger.info("Downloading dataset from Kaggle")
                api.dataset_download_files(self.config.source_URL, path=self.config.unzip_dir)
                logger.info("dataset download completed successfully")
            except Exception as e:
                logger.info('dataset download failed')
                logger.exception(e)
                raise e
        else:
            logger.info("Skipping download! Dataset already exists")
    
    def _get_updated_list_of_files(self,list_of_files):
        logger.info('Keep only .csv files from dataset zip file')
        return [f for f in list_of_files if f.endswith('.csv')]
    
    def _preprocess(self,zf:ZipFile,f:str,working_dir:str):
        target_filepath = os.path.join(working_dir,f)
        if not os.path.exists(target_filepath):
            zf.extract(f,working_dir)

        if os.path.getsize(target_filepath) == 0:
            logger.info(f"removing file {target_filepath} size: {get_size(target_filepath)}")
            os.remove(target_filepath)
            
    
    def unzip_and_clean(self):
        try:
            logger.info('unzipping dataset file started and removing 0 size files')
            with ZipFile(file=self.config.local_data_file,mode="r") as zf:
                list_of_files = zf.namelist()
                updated_list_of_files = self._get_updated_list_of_files(list_of_files)
                logger.info('pre-processing the dataset zip file and extraction starts')
                for f in tqdm(updated_list_of_files):
                    self._preprocess(zf,f,self.config.unzip_dir)
                logger.info('zip file extraction completed')
        except Exception as e:
            raise e