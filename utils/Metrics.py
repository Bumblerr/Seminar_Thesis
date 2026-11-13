from sklearn.metrics import (
    brier_score_loss, log_loss, roc_auc_score, 
    average_precision_score)

class Metrics:
    
    def __init__(self):
        pass
    @staticmethod
    def calculate_metrics(model_name, y_test, y_pred_prob):
        metrics = {
            "model_name": model_name,
            "brier_score": brier_score_loss(y_test, y_pred_prob),
            "log_loss": log_loss(y_test, y_pred_prob),
            "roc_auc": roc_auc_score(y_test, y_pred_prob),
            "pr_auc": average_precision_score(y_test, y_pred_prob),
        }
        return metrics
    
    @staticmethod
    def print_metrics(metrics):
        # Ausgabe
        print(f"   ✔ Brier-Score:              {metrics['brier_score']:.4f} -> optimal 0.0")
        print(f"   ✔ Log-Loss:                 {metrics['log_loss']:.4f} -> optimal 0.0")
        print(f"   ✔ ROC-AUC:                  {metrics['roc_auc']:.4f} -> optimal 1.0")
        print(f"   ✔ PR-AUC:                   {metrics['pr_auc']:.4f} -> optimal 1.0")
        
        
    @staticmethod
    def aggregate_cv_metrics(list_of_fold_metrics, model_name):
        """
        Aggregiert mehrere Fold-Metriken (Cross-Validation, Leave-One-Group-Out usw.).
        Erwartet eine Liste von Dicts, wie die Outputs von calculate_metrics().
        """
        import numpy as np

        # Mittelwert für jeden Zahlenwert außer model_name
        aggregated = {
            key: np.mean([fm[key] for fm in list_of_fold_metrics])
            for key in list_of_fold_metrics[0].keys()
            if key != "model_name"
        }

        aggregated["model_name"] = model_name
        return aggregated
