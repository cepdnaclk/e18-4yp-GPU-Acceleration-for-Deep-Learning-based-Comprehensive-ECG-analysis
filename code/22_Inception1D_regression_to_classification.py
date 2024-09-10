import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())
logging_enabled = False
import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import time
import utils.current_server as current_server
import numpy as np
import random
from sklearn.metrics import roc_auc_score
import logging

from torch.optim.lr_scheduler import ExponentialLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, fbeta_score

from models.Inception1D import Inception1d, Inception1dRegressionToClassification

from datasets.PTB_XL.PTB_XL_ECG_Dataset import ECGDataset, SHAPE_2D

from datasets.PTB_XL_Plus.PTB_XL_PLUS_ECG_Dataset import PTB_XL_PLUS_ECGDataset, HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER, SUB_DATASET_A, SUB_DATASET_B


logging.basicConfig(
    filename='log_regression_to_classification.log',  # Log file name
    filemode='a',        # Append mode (use 'w' for overwrite mode)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.DEBUG  # Log level (can be DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

SAVED_MODEL_PATHS = [
    "saved_models/22_Inception1D_regression.py_hr_20240905_211535_snowy-cherry-2",
    "saved_models/22_Inception1D_regression.py_qrs_20240724_220543_crimson-surf-81",
    "saved_models/22_Inception1D_regression.py_pr_20240725_130507_stellar-puddle-82",
    "saved_models/22_Inception1D_regression.py_qt_20240905_232817_gentle-universe-3",
]

PARAMETER_ORDER_LIST = [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER ]

for current_model_path, current_parameter in zip(SAVED_MODEL_PATHS, PARAMETER_ORDER_LIST):
    # Hyperparameters
    batch_size = 31
    learning_rate = 0.01
    num_epochs = 1000
    train_fraction = 0.8  # so test fraction is 0.2
    val_fraction = 0.1  # val fraction is 0.1 out of the total dataset | 0.125 out of train fraction
    select_sub_dataset = SUB_DATASET_B

    patience = 50
    early_stopping_counter = 0
    best_val_auc_roc = 0

    # Set a fixed seed for reproducibility
    SEED = 42

    # Set the seed for CPU
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # Set the seed for CUDA (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    source_model_name = current_model_path.split('_')[-1]
    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="deepfake_regression_to_ptbxl_classification", 
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": os.path.basename(__file__),
            "dataset": "PTB-XL-PLUS",
            "epochs": num_epochs,
            "parameter": current_parameter,
            "sub_dataset": select_sub_dataset
        },
        notes=f"frozen 1-7 transfer learning from {current_parameter} : based on {source_model_name} to classification  at: {current_time}",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained models with correct device mapping
    model1 = torch.load(current_model_path, map_location="cuda:0") 
    # model1 = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)

    # Freeze the first 7 layers of the model # 42 is based on the manual inspection of the model architecture
    count=0
    for param in model1.parameters():
        param.requires_grad = False
        count+=1
        if(count>=7):
            break

    # Remove the last layer (MLP) from the model
    model1.layers[1].pop(8)

    # Create the model
    model = Inception1dRegressionToClassification(model1)
    model = model.to(device)

    # Create the dataset class
    # dataset = ECGDataset(input_shape=SHAPE_2D, num_of_leads=8)
    dataset = PTB_XL_PLUS_ECGDataset(num_of_leads=8, sub_dataset=select_sub_dataset, is_classification=True, limit_dataset_for_testing=False)

    # Split the dataset into training and validation sets
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=1 - train_fraction, random_state=42, shuffle=True)
    train_indices, val_indices = train_test_split(train_indices, test_size=(val_fraction/train_fraction), random_state=42, shuffle=True)

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)

    # set num_workers
    if current_server.is_running_in_server():
        print(f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 4")
        num_workers = 4
    else:
        print(f"Running in {current_server.get_current_hostname()} server, Settings num_workers to 0")
        num_workers = 0

    # Create data loaders for training and validation
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    scheduler = ExponentialLR(optimizer, gamma=0.99)  # Set the gamma value for exponential decay

    logging.info(f"Starting transfer learning of {source_model_name} A:{current_parameter}>>to>>B:classification")
    try:
        for epoch in range(num_epochs):
            model.train()
            total_correct = 0
            total_samples = 0
            all_outputs = []
            all_labels = []

            for i, data in tqdm(
                enumerate(train_dataloader, 0),
                total=len(train_dataloader),
                desc=f"Training Epoch {epoch + 1}/{num_epochs}",
            ):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                if logging_enabled:
                    try:
                        print("label      |     output")
                        for round in range(batch_size):
                            print(labels[round], "  |  ", outputs[round])
                        print()
                    except Exception as e:
                        # Print the error message
                        print("An error occurred at print label and output:", e)

                # Calculate accuracy
                predicted = torch.argmax(outputs, 1)
                labels_max = torch.argmax(labels, 1)
                total_correct += (predicted == labels_max).sum().item()
                total_samples += labels.size(0)

                try:
                    # Store outputs and labels for AUC-ROC calculation
                    all_outputs.extend(outputs.detach().cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except:
                    pass

            # Calculate confusion matrix
            y_true = np.argmax(all_labels, axis=1)
            y_pred = np.argmax(all_outputs, axis=1)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

            print(f"Confusion Matrix for Train Epoch {epoch + 1}:")
            print(conf_matrix)

            train_conf_matrix_as_str = np.array2string(conf_matrix, separator=", ")

            # Compute accuracy
            train_accuracy = total_correct / total_samples

            # Compute AUC-ROC
            all_outputs = np.array(all_outputs)
            all_outputs = np.exp(all_outputs) / np.sum(np.exp(all_outputs), axis=1).reshape(-1, 1)

            # all_labels = np.concatenate(all_labels, axis=0)
            train_auc_roc = roc_auc_score(y_true, all_outputs, multi_class="ovr")

            # Log metrics
            print(f"Epoch: {epoch+1} train_accuracy: {train_accuracy}, train_auc_roc: {train_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")
            # Validation loop
            # Update learning rate scheduler
            scheduler.step()

            model.eval()
            total_correct = 0
            total_samples = 0
            all_outputs = []
            all_labels = []

            with torch.no_grad():
                for i, data in tqdm(
                    enumerate(val_dataloader, 0),
                    total=len(val_dataloader),
                    desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
                ):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)

                    # Calculate accuracy
                    predicted = torch.argmax(outputs, 1)
                    labels_max = torch.argmax(labels, 1)
                    total_correct += (predicted == labels_max).sum().item()
                    total_samples += labels.size(0)

                    try:
                        # Store outputs and labels for AUC-ROC calculation
                        all_outputs.extend(outputs.detach().cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())
                    except:
                        pass

                # Calculate confusion matrix
                y_true = np.argmax(all_labels, axis=1)
                y_pred = np.argmax(all_outputs, axis=1)
                conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

                print(f"Confusion Matrix for Val Epoch {epoch + 1}:")
                print(conf_matrix)

                val_conf_matrix_as_str = np.array2string(conf_matrix, separator=", ")

                # Compute accuracy
                val_accuracy = total_correct / total_samples

                # Compute AUC-ROC
                all_outputs = np.array(all_outputs)
                all_outputs = np.exp(all_outputs) / np.sum(np.exp(all_outputs), axis=1).reshape(-1, 1)

                val_auc_roc = roc_auc_score(y_true, all_outputs, multi_class="ovr")

                # Log metrics
                print(f"Epoch: {epoch+1} val_accuracy: {val_accuracy}, val_auc_roc: {val_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")

            #  Log metrics
            wandb.log(
                {
                    "train_accuracy": train_accuracy,
                    "train_AUC": train_auc_roc,
                    "val_accuracy": val_accuracy,
                    "val_AUC": val_auc_roc,
                    "lr": scheduler.get_last_lr()[0],
                    "train_confusion_matrix": train_conf_matrix_as_str,
                    "val_confusion_matrix": val_conf_matrix_as_str,
                }
            )
            # Early stopping
            if val_auc_roc > best_val_auc_roc:
                best_val_auc_roc = val_auc_roc
                best_model = model
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if early stopping criteria is met
            if early_stopping_counter >= patience:
                print(f"********Early stopping at epoch {epoch+1}********")
                break

        # end of trainging Start of Testing
        print("Using best model for Testing...")
        best_model.eval()
        total_correct = 0
        total_samples = 0
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for i, data in tqdm(
                enumerate(test_dataloader, 0),
                total=len(test_dataloader),
                desc=f"Testing",
            ):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = best_model(inputs)

                # Calculate accuracy
                predicted = torch.argmax(outputs, 1)
                labels_max = torch.argmax(labels, 1)
                total_correct += (predicted == labels_max).sum().item()
                total_samples += labels.size(0)

                try:
                    # Store outputs and labels for AUC-ROC calculation
                    all_outputs.extend(outputs.detach().cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                except:
                    pass
            # Calculate confusion matrix
            y_true = np.argmax(all_labels, axis=1)
            y_pred = np.argmax(all_outputs, axis=1)
            conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3, 4])

            print(f"Confusion Matrix for Val Epoch {epoch + 1}:")
            print(conf_matrix)

            test_conf_matrix_as_str = np.array2string(conf_matrix, separator=", ")

            # Compute accuracy
            test_accuracy = total_correct / total_samples

            # Compute AUC-ROC
            all_outputs_concat = np.array(all_outputs)
            # all_labels_concat = np.array(all_labels)
            # do softmax on all_outputs_concat row wise
            all_outputs_concat = np.exp(all_outputs_concat) / np.sum(np.exp(all_outputs_concat), axis=1).reshape(-1, 1)
            test_auc_roc = roc_auc_score(y_true, all_outputs_concat, multi_class="ovr")

            precision = precision_score(y_true, y_pred, average=None)

            recall = recall_score(y_true, y_pred, average=None)

            f1 = f1_score(y_true, y_pred, average=None)

            f0_5 = fbeta_score(y_true, y_pred, beta=0.5, average=None)

            f2 = fbeta_score(y_true, y_pred, beta=2, average=None)

            # Log metrics
            print(f"test_accuracy: {test_accuracy}, test_auc_roc: {test_auc_roc}, total_correct: {total_correct}, total_samples: {total_samples}")

        #  Log metrics
        wandb.log(
            {
                "test_accuracy": test_accuracy,
                "test_AUC": test_auc_roc,
                "test_confusion_matrix": test_conf_matrix_as_str,
                "confusion_matix": wandb.sklearn.plot_confusion_matrix(y_true, y_pred, dataset.labels),
                "precision_label0": precision[0],
                "precision_label1": precision[1],
                "precision_label2": precision[2],
                "precision_label3": precision[3],
                "precision_label4": precision[4],
                "recall_label0": recall[0],
                "recall_label1": recall[1],
                "recall_label2": recall[2],
                "recall_label3": recall[3],
                "recall_label4": recall[4],
                "f1_label0": f1[0],
                "f1_label1": f1[1],
                "f1_label2": f1[2],
                "f1_label3": f1[3],
                "f1_label4": f1[4],
                "f0_5_label0": f0_5[0],
                "f0_5_label1": f0_5[1],
                "f0_5_label2": f0_5[2],
                "f0_5_label3": f0_5[3],
                "f0_5_label4": f0_5[4],
                "f2_label0": f2[0],
                "f2_label1": f2[1],
                "f2_label2": f2[2],
                "f2_label3": f2[3],
                "f2_label4": f2[4],
            }
        )

        # Save the trained model with date and time in the path
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"saved_models/{os.path.basename(__file__)}_classification_{current_time}_{wandb.run.name}"

        torch.save(best_model, model_path)
        print("Finished Training")
        wandb.finish()
        logging.info(f"Ending transfer learning of {source_model_name} A:{current_parameter}>>to>>B:classification")
    except Exception as e:
        logging.error(f"Error in # and and the error is :{e}")
        print(f"Error in # and and the error is :{e}")
        try:
            wandb.finish(-1)
        except:
            logging.error("Error when tring to mark the wandb as crashed, qutting the whole program")
            print("Error when tring to mark the wandb as crashed, qutting the whole program")
            quit()
