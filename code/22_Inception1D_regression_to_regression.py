import utils.others as others

print(f"Last updated by: ", others.get_latest_update_by())
logging_enabled = False

import torch
import torch.nn as nn
from tqdm import tqdm
import datetime
import wandb
import os
import numpy as np
import random
import utils.current_server as current_server
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import r2_score
import logging

from models.Inception1D import Inception1d

from datasets.deepfake_ecg.Deepfake_ECG_Dataset import Deepfake_ECG_Dataset, CH_8_2D_MATRIX_OUTPUT_TYPE

from datasets.PTB_XL_Plus.PTB_XL_PLUS_ECG_Dataset import PTB_XL_PLUS_ECGDataset, HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER, SUB_DATASET_A, SUB_DATASET_B

logging.basicConfig(
    filename='log_regression_to_regression.log',  # Log file name
    filemode='a',        # Append mode (use 'w' for overwrite mode)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    level=logging.DEBUG  # Log level (can be DEBUG, INFO, WARNING, ERROR, CRITICAL)
)

SAVED_MODEL_PATHS = ["saved_models/22_Inception1D_regression.py_hr_20240620_074721_driven-night-43",
                     "saved_models/22_Inception1D_regression.py_qrs_20240620_095236_proud-shape-45",
                     "saved_models/22_Inception1D_regression.py_pr_20240620_112517_distinctive-violet-47",
                     "saved_models/22_Inception1D_regression.py_qt_20240620_130354_giddy-night-49"]

PARAMETER_ORDER_LIST = [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]

for current_model_path, current_parameter in zip(SAVED_MODEL_PATHS, PARAMETER_ORDER_LIST):
    # Hyperparameters
    batch_size = 31
    learning_rate = 0.01
    num_epochs = 1000
    train_fraction = 0.8  # so test fraction is 0.2
    val_fraction = 0.1  # val fraction is 0.1 out of the total dataset | 0.125 out of train fraction
    parameter = current_parameter
    select_sub_dataset = SUB_DATASET_B

    patience = 50
    early_stopping_counter = 0

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

    best_model = None
    best_validation_loss = 1000000
    
    source_model_name = current_model_path.split('_')[-1]

    # start a new wandb run to track this script
    wandb.init(
        # set the wandb project where this run will be logged
        project="subset_regression_to_regression",
        # track hyperparameters and run metadata
        config={
            "learning_rate": learning_rate,
            "architecture": os.path.basename(__file__),
            "dataset": "PTB XL PLUS",
            "epochs": num_epochs,
            "parameter": parameter,
            "sub_dataset": select_sub_dataset,
        },
        notes=f"4# freeze 1-7 out of 56 using {source_model_name} A:{parameter}>>to>>B:{parameter}",
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create the model / Load the model
    # model = Inception1d(num_classes=1, input_channels=8, use_residual=True, ps_head=0.5, lin_ftrs_head=[128], kernel_size=40).to(device)
    model = torch.load(current_model_path, map_location="cuda:0")

    # Freeze the first 7 layers of the model # 42 is based on the manual inspection of the model architecture
    count=0
    for param in model.parameters():
        param.requires_grad = False
        count+=1
        if(count>=7):
            break

    # Create the dataset class
    dataset = PTB_XL_PLUS_ECGDataset(parameter, num_of_leads=8, sub_dataset=select_sub_dataset)

    # Split the dataset into training and validation sets
    train_indices, test_indices = train_test_split(range(len(dataset)), test_size=1 - train_fraction, random_state=42, shuffle=True)
    train_indices, val_indices = train_test_split(train_indices, test_size=(val_fraction / train_fraction), random_state=42, shuffle=True)

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
    criterion = nn.L1Loss()
    scheduler = ExponentialLR(optimizer, gamma=0.99)  # Set the gamma value for exponential decay

    logging.info(f"Starting transfer learning of {source_model_name} A:{parameter}>>to>>B:{parameter}")
    # Training loop
    try:
        for epoch in range(num_epochs):
            training_constant_yes = 0
            number_of_constents_per_epoch = 0
            number_of_not_constents_per_epoch = 0
            number_of_constent_percentage = 0
            model.train()
            train_loss = 0.0
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

                try:
                    # Take the absolute value of the outputs
                    abs_outputs = torch.abs(outputs)

                    # Round up each element to the nearest integer
                    rounded_outputs = torch.ceil(abs_outputs)

                    # Convert to type Long
                    rounded_outputs = rounded_outputs.long()
                    flattened_rounded_outputs = rounded_outputs.view(-1)

                    # Count the occurrences of the most common element
                    most_common_count = flattened_rounded_outputs.bincount().max()

                    # Check if 25 elements out of 32 are the same
                    training_constant_yes = most_common_count >= batch_size - 7
                    if training_constant_yes:
                        number_of_constents_per_epoch += 1
                    else:
                        number_of_not_constents_per_epoch += 1

                    if logging_enabled:
                        print("same yes no", training_constant_yes)
                        print()
                except Exception as e:
                    # Print the error message
                    print("An error occurred at const number calc:", e)

                train_loss += loss.item()
            # Update learning rate
            scheduler.step()

            # Validation loop
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for i, data in tqdm(
                    enumerate(val_dataloader, 0),
                    total=len(val_dataloader),
                    desc=f"Validating Epoch {epoch + 1}/{num_epochs}",
                ):
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)

                    outputs = model(inputs)
                    # if i == 0:
                    #     for x in range(len(outputs)):
                    #         print(f"Predicted: {outputs[x]} Real: {labels[x]}")
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
            try:
                number_of_constent_percentage = (number_of_constents_per_epoch / (number_of_constents_per_epoch + number_of_not_constents_per_epoch)) * 100
            except Exception as e:
                # Print the error message here divide by zero error occured
                print("An error occurred at const percentage calc:", e)
            #  Log metrics
            wandb.log(
                {
                    "train_loss": train_loss / (len(train_dataloader)),
                    "val_loss": val_loss / (len(val_dataloader)),
                    "constant_percentage": number_of_constent_percentage,
                    "lr": scheduler.get_last_lr()[0],
                }
            )

            print(f"Epoch: {epoch} train_loss: {train_loss /  (len(train_dataloader))}")
            print(f"Epoch: {epoch} val_loss: {val_loss /  (len(val_dataloader))}")

            if (val_loss / (len(val_dataloader))) < best_validation_loss:
                best_validation_loss = val_loss / (len(val_dataloader))
                best_model = model
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            # Check if early stopping criteria is met
            if early_stopping_counter >= patience:
                print(f"********Early stopping at epoch {epoch}********")
                break

        # End of traning and start of Testing
        print("Using best model for testing...")
        best_model.eval()
        mae = 0.0
        all_labels = []
        all_outputs = []
        with torch.no_grad():
            for i, data in tqdm(
                enumerate(test_dataloader, 0),
                total=len(test_dataloader),
                desc=f"Testing",
            ):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = best_model(inputs)
                loss = criterion(outputs, labels)

                mae += loss.item()

                all_labels.extend(labels.cpu().numpy())
                all_outputs.extend(outputs.cpu().numpy())

        # Convert lists to numpy arrays
        all_labels = np.array(all_labels)
        all_outputs = np.array(all_outputs)

        # calculate MSE from all_labels and all_outputs
        mse = np.square(np.subtract(all_labels, all_outputs)).mean()
        # rmse
        rmse = np.sqrt(mse)
        # mape
        mape = np.mean(np.abs((all_labels - all_outputs) / all_labels)) * 100

        # r^2
        r_squared = r2_score(all_labels, all_outputs)

        # adjusted R^2
        n = len(all_labels)
        p = 1
        adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1)

        # median AE
        median_ae = np.median(np.abs(all_labels - all_outputs))

        # rmsle
        rmsle = np.sqrt(np.mean(np.power(np.log1p(all_labels) - np.log1p(all_outputs), 2)))

        # explained var
        explained_var = 1 - np.var(all_labels - all_outputs) / np.var(all_labels)

        #  Log metrics
        wandb.log(
            {
                "test_mae": mae / (len(test_dataloader)),
                "test_mse": mse,
                "test_rmse": rmse,
                "test_mape": mape,
                "test_r_squared": r_squared,
                "test_adjusted_r_squared": adjusted_r_squared,
                "test_median_ae": median_ae,
                "test_rmsle": rmsle,
                "test_explained_var": explained_var,
            }
        )

        print(f"Test_loss: {mae /  (len(test_dataloader))}")

        # Save the trained model with date and time in the path
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f"saved_models/{os.path.basename(__file__)}_{parameter}_{current_time}_{wandb.run.name}"

        torch.save(best_model, model_path)
        print("Best Model Saved")
        print("Finished Training")
        logging.info(f"Finished Traning of transfer learning {source_model_name} A:{parameter}>>to>>B:{parameter}")
        wandb.finish()
    except Exception as e:
        logging.error(f"Error in {source_model_name} A:{parameter}>>to>>B:{parameter} and the error is {e}")
        try:
            wandb.finish(-1)
        except:
            logging.error(f"Error when tring to mark the wandb as crashed")
            quit()
