import torch
import torch.nn as nn
import wandb
from utils import get_config, get_labels, add_jsonl
import pathlib
from dataset import get_dataloader
import tqdm
import sklearn.metrics
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
import numpy as np
import settings

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)  # Simple linear model.
        
    def forward(self, x):
        x = self.linear(x)
        return x

def train(files_dir: str, size_epoch = 10000, epochs: int=1, prohibited_classes: list=[], valid_no_class: bool=False, debugging: bool=False):
    # define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define transform functions (normalization)

    # get dataloaders
    train_path = pathlib.Path("datasets/train_dataset.hdf5")
    val_path = pathlib.Path("datasets/test-val_dataset.hdf5")
    test_path = pathlib.Path("datasets/test_dataset.hdf5")
    train_loader, val_loader, test_loader = get_dataloader(train_path, val_path, test_path, prohibited_classes = prohibited_classes, debugging = debugging)

    # get model
    input_dim = 512 # number of features in the input data
    output_dim = 168 # number of classes in the target
    model = LinearClassifier(input_dim, output_dim)
    model.to(device)

    # define loss_function
    criterion = torch.nn.CrossEntropyLoss()
    
    # define optimizer and learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # define proba function
    softmax = nn.Softmax(dim=1)  # The softmax function is ONLY used to compute 
                                 # scores for the missed_logos.json file. 

    # define lists for metrics
    y_train = []
    y_test = []
    y_pred = []
    classes_str, classes_ids = get_labels(settings.labels_path, prohibited_classes=prohibited_classes)

    # training loop
    for epoch in range(epochs):
        count = 0
        model.train()
        train_loss = 0.0
        print("Entering training loop")
        for embeddings, labels, ids in tqdm.tqdm(train_loader):
            count += 1
            optimizer.zero_grad()
            outputs = model.forward(embeddings.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            ground_truth = torch.tensor([torch.where(classe==1)[0] for classe in labels])
            y_train += ground_truth
            if count >= size_epoch:
                break

        # Evaluate the model on validation set
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        missed_no_logos = {}
        y_test = []
        y_pred = []
        with torch.no_grad():
            print("Entering testing loop")
            for embeddings, labels, ids in tqdm.tqdm(val_loader):
                outputs = model.forward(embeddings.to(device))
                loss = criterion(outputs, labels.to(device))
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                ground_truth = torch.tensor([torch.where(classe==1)[0] for classe in labels])
                correct += (predicted == ground_truth.to(device)).sum().item()
                scores = softmax(outputs)
                for indice in torch.where((predicted == ground_truth.to(device))==False)[0].tolist():
                    missed_no_logos[str(ids[indice].item())]=[ground_truth[indice].item(), predicted[indice].item(), scores[indice][predicted[indice]].item()]

                if not valid_no_class:  # To compute the metrics, we take out all no_class logos 
                                        # (not the ones predicted as no_class but the true no_class)                
                    predicted = predicted[torch.where(ground_truth!=0)]
                    ground_truth = ground_truth[torch.where(ground_truth!=0)]
                y_pred += predicted.tolist()
                y_test += ground_truth.tolist()

        # Compute metrics and save model
        current_macro_f1 = compute_metrics(y_test, y_pred, classes_ids, classes_str, len(train_loader), len(val_loader), correct, total, train_loss, val_loss, epoch)
        
        save_best_model = SaveBestModel()
        save_best_model(current_macro_f1, missed_no_logos, epoch, model, device, files_dir)

def compute_metrics(y_test, y_pred, classes_ids, classes_str, len_train_loader, len_val_loader, correct, total, val_loss, train_loss, epoch):

    # Compute all metrics. Report contains all relevant data.
    report = sklearn.metrics.classification_report(y_test, y_pred, labels=classes_ids, target_names=classes_str, zero_division=0)
    f1_micro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="micro")
    f1_macro = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average="macro", zero_division=0)
    f1_classes = sklearn.metrics.f1_score(y_true=y_test, y_pred=y_pred, labels = classes_ids, average=None, zero_division=0)
    total_accuracy = 100 * correct / total
    training_loss = train_loss / len_train_loader
    validation_loss = val_loss / len_val_loader

    metrics_dict = {
        "epoch": epoch + 1,
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "total_accuracy": total_accuracy,
        "training_loss": training_loss,
        "validation_loss": validation_loss,
        }
    
    for i in range(len(f1_classes)):
        metrics_dict["f1 by class/f1_" + str(classes_str[i])] = f1_classes[i]
    wandb.log(metrics_dict)

    # The confusion matrix is displayed in wandb.
    confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)
    confusion_matrix = normalize(confusion_matrix, axis=1)

    print(f"report: {report}")
    print(f"confusion_matrix: {confusion_matrix}")
    fig = plt.figure(figsize=(24, 20))

    plt.imshow(confusion_matrix, cmap=plt.get_cmap("Blues"))
    ax = plt.gca()
    ax.set_xticks(np.arange(-.5, len(classes_str)-1, 1))
    ax.set_yticks(np.arange(-.5, len(classes_str)-1, 1))
    ax.set_xticklabels(
        classes_str,
        size='smaller',
        rotation='vertical',
        )
    ax.set_yticklabels(
        classes_str,
        size='smaller',
        rotation='horizontal',
        )
    ax.grid(color='black', linestyle='-', linewidth=1)
    plt.colorbar()
    plt.ylabel("Predicted")
    plt.xlabel("True")
    wandb.log({"plot":fig})

    return f1_macro


class SaveBestModel:
    """
    Class to save the best model and the wrong_predictions while training. 
    If the current epoch's macro f1 is better than the previous best, 
    then save the model state.
    """
    def __init__(
        self, best_macro_f1=-float('inf')
    ):
        self.best_macro_f1 = best_macro_f1
        
    def __call__(
        self, current_macro_f1, missed_logos,
        epoch, model, device,
        dir_path, batch_size = 32
    ):
        if current_macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = current_macro_f1
            print(f"\nBest macro f1: {self.best_macro_f1}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.onnx.export(
                model,
                torch.zeros([batch_size, 512]).to(device),
                dir_path+'/logos_classifier.onnx',
                export_params=True,
                verbose=True,
                input_names = ['embeddings'],
                output_names = ['scores_per_classes'],
                dynamic_axes = {'embeddings':[0]}
                )
            add_jsonl([missed_logos], dir_path+'/missed_logos.json')

if __name__ == '__main__':
    '''
    Script used to train the logos classifier model.

    Give a name to the wandb project.
    Check the settings.py file for each file and parameters to be well defined.
    Configure the parameters of the train function:
        * size_epoch: amount of batches corresponding to one epoch of the training.
        * epochs: amount of epochs run for training
        * prohibited_classes: list of the ids of classes you want to prohibit for training, validation and test
        * valid_no_class: True if you want no_class logos to be taking into account for the computation 
        of metrics during validation step. False else.
        * debugging: True if you want to run the script by using test dataset as train and val, as it is shorter
        than the former. False for usual use.
    Run the script !
    '''


    run = wandb.init(project="with_no_class-logos-classifier")
    files_dir = run.dir
    train(files_dir=files_dir, size_epoch=1680, epochs=200, prohibited_classes=[], valid_no_class=True, debugging=False)
    run.finish()
