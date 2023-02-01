import os,shutil,re
import urllib.request as request
from zipfile import ZipFile
import json
from kaggle.api.kaggle_api_extended import KaggleApi
from waferFaultDetection import logger
from waferFaultDetection.constants import *
from waferFaultDetection.entity import DataIngestionConfig
from waferFaultDetection.utils import get_size
from tqdm import tqdm
import pandas as pd

class DataIngestion:
    '''
    1) Downloads data file from Kaggle
    2) Unzips the file and cleans
    3) Creates schema
    4) Validates the input data for file names and columns
    5) Creates modle input file
    '''
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

    def _create_schema_train(self):
        root_dir = self.config.root_dir
        schema_train_dict = {
            "SampleFileName":"wafer_31122020_000000.csv",
            "LengthOfDateStamp":8,
            "LengthOfTimeStamp":6,
            "NumOfCols":592,
            "Columns":{}
        }
        numCols = schema_train_dict['NumOfCols']
        for i in tqdm(range(numCols)):
            if i == 0:
                schema_train_dict['Columns']['wafer'] = "str"
            elif i == numCols - 1:
                schema_train_dict['Columns']['Output'] = "int"
            else:
                col = "Sensor - "+str(i)
                schema_train_dict['Columns'][col] = "float"
        json_out = os.path.join(root_dir,"schema_train.json")
        # with open(json_out,"w") as outfile:
        #     json.dump(schema_train_dict,outfile)
        return schema_train_dict
        

    def validate_file_names(self):
        root_dir = self.config.root_dir
        # dirpath,dirnames,list_of_files = os.walk(root_dir)
        logger.info("Finding the training files directory")
        try:
            dirs = [f for f in os.listdir(root_dir) if 'train' in f.lower()]
            if len(dirs) == 1:
                train_dir = os.path.join(root_dir,dirs[0])
            else:
                raise ValueError(print(dirs))
            logger.info("Finding the training data with .csv extension")
            list_of_files = [f for f in os.listdir(train_dir) if f.endswith('.csv')]
            good_files_dir = Path(os.path.join(train_dir,"good_files_dir"))
            bad_files_dir = Path(os.path.join(train_dir,"bad_files_dir"))
            logger.info("Creating good and bad files directories")
            if not os.path.exists(good_files_dir):
                os.makedirs(good_files_dir)
            if not os.path.exists(bad_files_dir):
                os.makedirs(bad_files_dir)
            logger.info("Validation of file names started >>>>>>>>>>>>>>>")
            logger.info("Compare file names and move them to good or bad files directory")
            regex = 'wafer_'+"\d{8}"+"_"+"\d{6}"+".csv"
            for file in list_of_files:
                # dir,fname = os.path.split(file)
                if re.match(regex,file.lower()):
                    shutil.move(os.path.join(train_dir,file),os.path.join(good_files_dir,file))
                else:
                    shutil.move(os.path.join(train_dir,file),os.path.join(bad_files_dir,file))
            logger.info("Validation of file names completed >>>>>>>>>>>>>>>")
            self.train_dir = train_dir
            self.good_files_dir = good_files_dir
            self.bad_files_dir = bad_files_dir
        except OSError:
            logger.info("OSError: Error validating file names and moving to bad_files_dir")
        except Exception as e:
            logger.exception(e)
            raise e

    def validate_columns(self):
        good_files_dir = self.good_files_dir
        bad_files_dir = self.bad_files_dir
        logger.info("create train data schema")
        schema_train_dict = self._create_schema_train()
        logger.info("<<<<<<<<<<<<<< Validation of number of columns started >>>>>>>>>>>>>>>")
        logger.info("checking the number of columns in each file and moving to bad_files_dir if not met")
        try:
            for file in tqdm(os.listdir(good_files_dir)):
                # logger.info(f"pandas reading file: {file} in {good_files_dir}")
                df = pd.read_csv(os.path.join(good_files_dir,file))
                if not len(df.columns) == schema_train_dict['NumOfCols']:
                    logger.info(f"file: {file} moving into the bad_files_dir: length of columns: {len(df.columns)} < {schema_train_dict['NumOfCols']}")
                    shutil.move(os.path.join(good_files_dir,file),os.path.join(bad_files_dir,file))
                else:
                    df.fillna('NULL',inplace=True)
                    df_null = (df != 'NULL')
                    # dfnull = df_null.sum(axis=0)
                    dfnull = (df_null.sum(axis=0) == 0)
                    if len(dfnull[dfnull == True].index) != 0:
                        logger.info(f"file: {file} moving into the bad_files_dir: file contains whole columns with missing values")
                        shutil.move(os.path.join(good_files_dir,file),os.path.join(bad_files_dir,file))
                    else:
                        # create csv files from df with updated NULL values
                        df.rename(columns={"Unnamed: 0": "Wafer"}, inplace=True)
                        fname = os.path.join(good_files_dir,file)
                        df.to_csv(fname,index=False,header=True)
            logger.info("<<<<<<<<<<<<<< Validation of number of columns completed >>>>>>>>>>>>>>>")
        except OSError:
            logger.info("OSError: Error validating columns and moving to bad_files_dir")
        except Exception as e:
            logger.exception(e)
            raise e
                    

    def create_model_input_file(self):
        logger.info("****************** creation of ML model input file started *******************")
        good_files_dir = self.good_files_dir
        model_input_file = self.config.model_input_file
        df = pd.DataFrame()
        for file in os.listdir(good_files_dir):
            file = os.path.join(good_files_dir,file)
            df_csv = pd.read_csv(file)
            df = pd.concat([df,df_csv],axis=0,ignore_index=True)
        else:
            df.to_csv(model_input_file,index=False)
            logger.info("****************** creation of ML model input file completed *******************")