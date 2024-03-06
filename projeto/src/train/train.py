import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))

import pandas as pd
import structlog
import pickle

logger = structlog.get_logger()

from utils.utils import Utils

my_utils = Utils()



class TrainModels:
    """
    Classe responsável por treinar modelos de classificação.
    """

    def __init__(self, dados_x: pd.DataFrame, dados_y: pd.DataFrame):
        """
        Inicializa a classe TrainModels.

        Parâmetros:
        - dados_x: DataFrame contendo os dados de entrada para treinamento.
        - dados_y: DataFrame contendo os rótulos correspondentes aos dados de entrada.
        """
        self.dados_x = dados_x
        self.dados_y = dados_y
        self.model_name = my_utils.load_config_file().get('model_name')

    def train(self, model):
        """
        Treina o modelo de classificação.

        Parâmetros:
        - model: O modelo de classificação a ser treinado.

        Retorna:
        - O modelo treinado.
        """
        model.fit(self.dados_x, self.dados_y)
        my_utils.save_model(model, self.model_name)
        return model

    
