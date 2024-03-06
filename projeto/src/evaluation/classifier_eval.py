import structlog
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, stratifiedKFold

from utils.utils import Utils

my_utils = Utils()

logger = structlog.get_logger()

class ModelEvaluation:
    def __init__(self, model, X, y, cv=5):
        self.model = model
        self.X = X
        self.y = y
        self.cv = cv

    def cross_val_evaluate(self):
        logger.info("Iniciando a validação cruzada...")
        scores = cross_val_score(self.model, self.X, self.y, cv=self.cv, scoring="roc_auc")
        skf = StratifieldKFold(n_splits=self.cv, 
                               shuffle=True, 
                               random_state=my_utils.load_config_file().get('random_state'))
        scores = cross_val_score(self.model, self.X, self.y, cv=skf, scoring="roc_auc")
        return scores

    def roc_auc_scorer(self):
        y_pred_proba = self.model.predict_proba(self.X)[:,1]
        return roc_auc_score(self.y, y_pred)

    @staticmethod
    def evaluate_prediction(y_true, y_pred):
        logger.info("Avaliando as predições...")
        return roc_auc_score(y_true, y_pred)



