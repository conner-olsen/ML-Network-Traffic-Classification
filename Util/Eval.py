

from sklearn.metrics import accuracy_score, \
    confusion_matrix, \
    precision_score, \
    recall_score, \
    f1_score

from Util.Util import get_results_location


def evaluate_model(x_test, y_test, models_folder, model_name: str, predictions):
    """
    Test the given model on the given data, and write the
    results to a file.

    :param x_test: Data
    :param y_test: Labels
    :param models_folder: Model
    :param model_name: Name
    :param predictions: returned from model.predict()
    :return: Nothing
    """

    # Compute evaluation metrics
    metrics = {
        'Accuracy': accuracy_score(y_test, predictions),
        'Precision': precision_score(y_test, predictions, average='micro'),
        'Recall': recall_score(y_test, predictions, average='micro'),
        'F1 Score': f1_score(y_test, predictions, average='micro'),
        'Confusion Matrix': confusion_matrix(y_test, predictions)
    }

    # Write the evaluation metrics to a file
    with open(get_results_location(models_folder, model_name), 'a') as f:
        for metric, value in metrics.items():
            f.write(f'{metric}: {value}\n')

    # Also print the results to stdout
    for metric, value in metrics.items():
        print(f'{metric}: {value}')
