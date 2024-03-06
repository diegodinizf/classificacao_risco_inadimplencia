import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import structlog
from utils.utils import Utils

my_utils = Utils()

import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import structlog

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

class DataTransformation:
    """
    Classe responsável por transformar os dados.
    """

    def __init__(self, dataframe: pd.DataFrame):
        """
        Inicializa a classe DataTransformation.

        Args:
            dataframe (pd.DataFrame): O dataframe contendo os dados.
        """
        self.dataframe = dataframe
        self.target_name = my_utils.load_config_file().get('target_name')

    def column_name_change(self):
        """
        Altera os nomes das colunas do dataframe para snake_case.

        Returns:
            pd.DataFrame: O dataframe com os nomes das colunas alterados.
        """
        dataframe_formatted_columns = self.dataframe.rename(columns=lambda x: my_utils.to_snake_case(x))
        return dataframe_formatted_columns

    def feature_engineering(self, dataframe_imported: pd.DataFrame) -> pd.DataFrame:
        """
        Realiza a engenharia de features no dataframe.

        Args:
            dataframe_imported (pd.DataFrame): O dataframe importado.

        Returns:
            pd.DataFrame: O dataframe com as features engenheiradas.
        """
        # definindo uma faixa para a renda mensal
        dataframe_imported['faixa_renda_mensal'] = dataframe_imported['renda_mensal'].apply(lambda x: 'não especificado' if np.isnan(x) else
                                                                        '0-1k' if x >= 0 and x < 1000 else 
                                                                        '1k-5k' if x>=1000 and x < 5000 else
                                                                        '5k-20k' if x >= 5000 and x < 20000 else
                                                                        '20k-100k' if x>=20000 and x < 100000 else
                                                                        '100k-1mi' if x>=100000 and x < 1000000 else
                                                                        '1mi+'
                                                                        )

        # definindo faixa para numero de dependentes
        dataframe_imported['faixa_numero_dependentes'] = dataframe_imported['numero_de_dependentes'].apply(lambda x: 'não especificado' if np.isnan(x) else
                                                                                    '0-2' if x>=0 and x<2 else
                                                                                    '2-5' if x>=2 and x<5 else
                                                                                    '5-10' if x>=5 and x<10 else
                                                                                    '10+')
        
        dataframe_imported.drop(columns=['renda_mensal','numero_de_dependentes'], inplace = True)

        return dataframe_imported
    
    def train_test_splitting(self, featured_dataframe)->pd.DataFrame:
        """
        Realiza a divisão dos dados em treino e teste.

        Args:
            featured_dataframe (pd.DataFrame): O dataframe com as features engenheiradas.

        Returns:
            pd.DataFrame: Os dataframes de treino e teste.
        """
        X = featured_dataframe.drop(self.target_name, axis = 1)
        y = featured_dataframe[self.target_name]

        X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                                    test_size = my_utils.load_config_file().get('test_size') , 
                                                                    stratify=y, 
                                                                    random_state=my_utils.load_config_file().get('random_state')
                                                                    )

        return X_train, X_valid, y_train, y_valid

    def run(self):
        """
        Executa as transformações nos dados.

        Returns:
            pd.DataFrame: Os dataframes de treino e teste.
        """
        dataframe_formatted_columns = self.column_name_change()
        featured_dataframe = self.feature_engineering(dataframe_formatted_columns)
        X_train, X_valid, y_train, y_valid = self.train_test_splitting(featured_dataframe)

        return X_train, X_valid, y_train, y_valid