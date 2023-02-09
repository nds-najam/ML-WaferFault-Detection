from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    '''
    Define the resources required for the data ingestion
    such as URL, dataset file name, directory details
    '''
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    model_input_file: Path

@dataclass(frozen=True)
class DataPreprocessingConfig:
    '''
    Define the resources required for the data preprocessing
    such as dataset file name, label column name, null data summary, standard deviation summary etc.
    '''
    root_dir: Path
    model_input_file: Path
    sensor_name_column: str
    label_column_name: str
    null_summary_file: Path
    zero_stddev_columns_file: Path
    preprocessed_model_input_file: Path
    elbow_plot_file: Path
    kmeans_model_file: Path