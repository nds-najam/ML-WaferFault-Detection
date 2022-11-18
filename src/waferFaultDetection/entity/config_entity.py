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