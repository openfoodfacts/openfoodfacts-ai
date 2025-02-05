import torch
import tqdm
from train import SaveBestModel, compute_metrics
from dataset import get_dataloader
import pathlib
from utils import get_labels
import onnxruntime as ort
import numpy as np
import sklearn
import json
from math import inf
import matplotlib.pyplot as plt
import settings

def test_model(ort_session, prohibited_classes, test_dataset):

    train_path = pathlib.Path("datasets/train_dataset.hdf5")
    val_path = pathlib.Path("datasets/val_dataset.hdf5")
    test_path = pathlib.Path(test_dataset)
    _, _, test_loader = get_dataloader(train_path, val_path, test_path, prohibited_classes = prohibited_classes, test = True)

    classes_str, classes_ids = get_labels("datasets/class_infos.jsonl", prohibited_classes=prohibited_classes)

    # Define the probability function
    softmax = torch.nn.Softmax(dim=1)

    # Evaluate the model on validation set

    correct = 0
    total = 0
    missed_logos = {}
    details_predictions = {}
    y_test = []
    y_pred = []
    with torch.no_grad():
        print("Entering testing loop")
        for embeddings, labels, ids in tqdm.tqdm(test_loader):
            outputs = ort_session.run(None,{"embeddings":embeddings.detach().cpu().numpy()})[0]
            outputs = torch.Tensor(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            ground_truth = torch.tensor([torch.where(classe==1)[0] for classe in labels])
            correct += (predicted == ground_truth).sum().item()
            scores = softmax(outputs)
            for indice in range(len(predicted)):
                true = ground_truth[indice].item()
                prediction = predicted[indice].item()
                id = ids[indice].item()
                score = scores[indice][predicted[indice]].item()
                if predicted[indice] != ground_truth[indice]:
                    missed_logos[str(id)]=[true, prediction]
                if str(prediction) not in details_predictions.keys():
                    details_predictions[str(prediction)] = [[id, score, str(true)]]
                else:
                    details_predictions[str(prediction)].append([id, score, str(true)]) 

            y_pred += predicted.tolist()
            y_test += ground_truth.tolist()

    # Compute metrics 
    report = sklearn.metrics.classification_report(y_test, y_pred, labels=classes_ids, target_names=classes_str)
    print(report)

    return details_predictions

def compute_metrics_threshold(details_predictions: dict, thresholds: list):
    classes_str, classes_ids = get_labels(settings.labels_path, [])
    classes_str = np.array(classes_str)
    classes_ids = np.array(classes_ids)

    metrics = {}
    for predicted in tqdm.tqdm(details_predictions.keys()):
        if predicted == 0:
            continue
        predicted_str = classes_str[np.where(classes_ids==int(predicted))][0]
        metrics[predicted_str] = [[],[]]
        for threshold in thresholds:
            ground_truth = []
            prediction = []
            for _, score, truth in details_predictions[predicted]:
                if score >= threshold:
                    prediction.append(1)
                else:
                    prediction.append(0)
                if truth == predicted:
                    ground_truth.append(1)
                else:
                    ground_truth.append(0)
            recall = sklearn.metrics.recall_score(ground_truth, prediction)
            precision = sklearn.metrics.precision_score(ground_truth, prediction, zero_division = 1)
            metrics[predicted_str][0].append(recall)
            metrics[predicted_str][1].append(precision)
    return metrics


def find_best_threshold(thresholds: list, metrics: dict, min_precision: int):
    best_threshold = {}
    former_recall = [inf,0]
    for predicted in metrics.keys():
        i = 0
        while i+1<len(metrics[predicted][1]) and metrics[predicted][1][i+1]>=min_precision:
            i += 1
            if metrics[predicted][0][i] != former_recall[0]:
                former_recall = [metrics[predicted][0][i], i]
        best_threshold[predicted]={"precision":metrics[predicted][1][former_recall[1]], "recall":metrics[predicted][0][former_recall[1]], "threshold": thresholds[former_recall[1]]}
    return best_threshold

def save_ROC(metrics: dict):
    for predicted in metrics.keys():
        plt.plot(metrics[predicted][0], metrics[predicted][1])
        plt.title("Precision Recall Curve for "+predicted)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("ROC_curves/" + predicted + ".png")
        plt.clf()


if __name__ == "__main__":
    '''
    This script is used to test the onnx classifier model trained with train.py script.
    You can choose the dataset you want to test it on through the parameter "test_dataset".
    '''
    test_dataset = "datasets/test-val_dataset.hdf5"
    model = "logos_classifier.onnx"
    ort_session = ort.InferenceSession(model, providers=['CUDAExecutionProvider'])
    details_predictions = test_model(ort_session=ort_session, prohibited_classes=[], test_dataset)

    total = 100
    thresholds = [round((1-(i+1)/total),2) for i in range(total)]
    metrics = compute_metrics_threshold(details_predictions, thresholds)
    best_thresholds = find_best_threshold(thresholds, metrics, min_precision = 0.98)
    save_ROC(metrics)
    with open('threshold_metrics.json', 'w') as f:
        f.write(json.dumps(best_thresholds, indent = 3))

