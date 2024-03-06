import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
from utils.utils import Utils
import structlog

logger = structlog.get_logger()

my_utils = Utils()

class DataLoad:
    """Classe para carregamento de dados"""
    def __init__(self) -> None:
        pass

    def load_data(self, dataset_name: str) -> pd.DataFrame:
        """Carrega os dados a partir do nome do dataset fornecido
        
        Args:
            dataset_name (str): Nome do dataset a ser carregado
        
        Retorna:
            pandas DataFrame: Dados carregados

        Raises:
            ValueError: Se o dataset não for encontrado no arquivo de configuração
            Exception: Erro inesperado
        """

        logger.info(f"Carregando o dataset {dataset_name}...")

        # Carrega os dados do arquivo CSV
        try:
            dataset = my_utils.load_config_file().get(dataset_name)
            if dataset is None:
                raise ValueError(f"Dataset {dataset_name} não encontrado no arquivo de configuração.")

            # Retorna os dados carregados
            loaded_data = pd.read_csv(f'../data/raw/{dataset}')

            return loaded_data[my_utils.load_config_file().get('columns_to_use')]

        except ValueError as ve:
            logger.error(str(ve))

        except Exception as e:
            logger.error(f'Erro inesperado: {str(e)}')