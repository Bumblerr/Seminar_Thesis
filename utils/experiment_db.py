import sqlite3


"""Muss immer angepasst werden, wenn neue Metriken hinzukommen."""

def init_db(db_path="results/experiments.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_results (
        model_name TEXT PRIMARY KEY,
        brier_score REAL,
        log_loss REAL,
        roc_auc REAL,
        pr_auc REAL
    )
    """)
    conn.commit()
    conn.close()
    

def log_or_update_metrics(model_name, metrics, db_path="results/experiments.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO model_results 
        (model_name, brier_score, log_loss, roc_auc, pr_auc)
    VALUES 
        (:model_name, :brier_score, :log_loss, :roc_auc, :pr_auc)
    ON CONFLICT(model_name)
    DO UPDATE SET
        brier_score=excluded.brier_score,
        log_loss=excluded.log_loss,
        roc_auc=excluded.roc_auc,
        pr_auc=excluded.pr_auc;
    """, metrics)

    conn.commit()
    conn.close()

