import os
import yaml
import re
from sklearn.base import BaseEstimator, TransformerMixin
import pickle


class Utils:
    """
    Classe contendo funções utilitárias.
    """

    @staticmethod
    def load_config_file():
        """
        Carrega o arquivo de configuração e retorna seu conteúdo.

        Returns:
            dict: O conteúdo do arquivo de configuração.
        """
        # Obtém o diretório atual do arquivo
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))

        # Define o caminho relativo para o arquivo de configuração
        caminho_relativo = os.path.join('..', '..', 'config', 'config.yaml')

        # Obtém o caminho absoluto para o arquivo de configuração
        config_file_path = os.path.abspath(os.path.join(diretorio_atual, caminho_relativo))

        # Carrega o arquivo de configuração e retorna seu conteúdo
        config_file = yaml.safe_load(open(config_file_path, 'rb'))
        return config_file

    @staticmethod
    def to_snake_case(column_name):
        """
        Converte um nome de coluna para snake_case.

        Argumentos:
        column_name (str): O nome da coluna a ser convertido.

        Retorna:
        str: O nome da coluna convertido para snake_case.
        """
        # Converte letras maiúsculas para minúsculas e adiciona um underscore antes delas
        snake_case_name = re.sub(r'(?<!^)(?=[A-Z])', '_', column_name).lower()

        return snake_case_name

    def save_model(self, model, model_name):
        """
        Salva o modelo treinado.

        Argumentos:
        model: O modelo treinado.
        model_name (str): O nome do modelo.
        """
        # Obtém o diretório atual do arquivo
        diretorio_atual = os.path.dirname(os.path.abspath(__file__))

        # Define o caminho relativo para o modelo
        caminho_relativo = os.path.join('..', '..', 'models', model_name)

        # Obtém o caminho absoluto para o modelo
        model_path = os.path.abspath(os.path.join(diretorio_atual, caminho_relativo))

        # Salva o modelo
        with open(model_path, 'wb') as model_file:
            pickle.dump(model, model_file)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Classe para realizar o encoding de frequência em colunas específicas.
    """
    def __init__(self, columns):
        """
        Inicializa o FrequencyEncoder com as colunas especificadas.

        Argumentos:
        columns (list): Uma lista de nomes de colunas a serem encodadas.
        """
        self.columns = columns
        self.encoding_maps = {}

    def fit(self, data, y = None):
        """
        Ajusta o encoder com base nos dados fornecidos.

        Argumentos:
        data (pandas.DataFrame): Os dados de treinamento.
        """
        for column in self.columns:
            self.encoding_maps[column] = data[column].value_counts(normalize=True)
        return self

    def transform(self, data):
        """
        Aplica o encoding de frequência aos dados fornecidos.

        Argumentos:
        data (pandas.DataFrame): Os dados a serem encodados.

        Retorna:
        pandas.DataFrame: Os dados encodados.
        """
        encoded_data = data.copy()
        for column in self.columns:
            encoded_data[column] = data[column].map(self.encoding_maps[column])
        return encoded_data
