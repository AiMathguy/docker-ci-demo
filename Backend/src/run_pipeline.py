import logging

from src.train_model import ChurnModelManager

logging.basicConfig(level=logging.INFO)


def train_pipeline():
    manager = ChurnModelManager()
    manager.fetch_data()
    manager.inject_simulated_labels()
    results = manager.run_training_pipeline()

    logging.info(f"ROC-AUC: {results['auc']:.4f}")
    logging.info(f"Report:\n{results['report']}")

    return results


if __name__ == "__main__":
    train_pipeline()
