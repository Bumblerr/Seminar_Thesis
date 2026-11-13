from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


class Models:
    def __init__(self):
        pass
    
    def define_models():
        models = {
            'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
            
            'DecisionTree': DecisionTreeClassifier(
                max_depth=10,
                min_samples_leaf=30,
                criterion= 'gini',
                class_weight= 'balanced',
                random_state=42),

            'xgboost': XGBClassifier(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=5,
                random_state=42,
                eval_metric="logloss",
            )
        }
        
        return models