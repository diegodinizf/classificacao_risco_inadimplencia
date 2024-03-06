import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
import pandera
from pandera import Check, Column, DataFrameSchema, errors, error_handlers

import structlog

from utils.utils import Utils

my_utils = Utils()

logger = structlog.get_logger()

class DataValidation:
    """Classe para validação de dados."""

    def __init__(self) -> None:
        self.columns_to_use = my_utils.load_config_file().get('columns_to_use')

    def check_shape_data(self, dataframe: pd.DataFrame) -> bool:
        """Verifica a forma dos dados.

        Args:
            dataframe (pd.DataFrame): O dataframe de entrada.

        Returns:
            bool: True se a forma for válida, False caso contrário.
        """
        try:
            logger.info('Validação iniciada')
            dataframe.columns = self.columns_to_use
            return True
        except Exception as e:
            logger.error(f'Validação falhou: {e}')
            return False
    
    def check_columns(self, dataframe: pd.DataFrame) -> bool:
        """Verifica as colunas dos dados.

        Args:
            dataframe (pd.DataFrame): O dataframe de entrada.

        Returns:
            bool: True se as colunas forem válidas, False caso contrário.
        """
        schema = DataFrameSchema(
            {
             'target': Column(int, Check.isin([0,1]), Check(lambda x: x > 0), coerce = True),   
             'TaxaDeUtilizacaoDeLinhasNaoGarantidas': Column(float, nullable = True),
             'Idade': Column(int, nullable = True),
             'NumeroDeVezes30-59DiasAtrasoNaoPior': Column(int, nullable = True),
             'TaxaDeEndividamento': Column(float, nullable = True),
             'RendaMensal': Column(float, nullable = True),
             'NumeroDeLinhasDeCreditoEEmprestimosAbertos': Column(int, nullable = True),
             'NumeroDeVezes90DiasAtraso': Column(int, nullable = True),
             'NumeroDeEmprestimosOuLinhasImobiliarias': Column(int, nullable = True),
             'NumeroDeVezes60-89DiasAtrasoNaoPior': Column(int, nullable = True),
             'NumeroDeDependentes': Column(float, nullable = True)
            }
        )
        
        try:
            schema.validate(dataframe)
            logger.info("Column validation passed...")
            return True
        except pandera.errors.SchemaErrors as exc:
            logger.error("Column validation failed...")
            pandera.display(exc.failure_cases)
            return False
        
    def run(self, dataframe: pd.DataFrame) -> bool:
        """Run the data validation.

        Args:
            dataframe (pd.DataFrame): The input dataframe.

        Returns:
            bool: True if the data is valid, False otherwise.
        """
        if self.check_shape_data(dataframe) and self.check_columns(dataframe):
            logger.info("Validation successful")
            return True
        else:
            logger.error("Validation failed")
            return False