import torch
import pandas
from scipy.signal import spectrogram
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import torchvision.transforms.functional as F

import socket

# decide to run the full dataset or no based on the server or local machine
hostname = socket.gethostname()

server_hostnames = ["ampere", "turing", "aiken"]
if hostname in server_hostnames:
    # run full dataset on servers
    IS_FULL_DATASET = True
    print("Running full dataset")
else:
    # limit to 1000 on local computers
    IS_FULL_DATASET = False
    print("Running limited dataset")

HR_PARAMETER = "hr"
QRS_PARAMETER = "qrs"
PR_PARAMETER = "pr"
QT_PARAMETER = "qt"

DEFAULT_OUTPUT_TYPE = "default"
DEFAULT_SPECTROGRAM_OUTPUT_TYPE = "spectrogram"
VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE = "vision_transformer_image"
VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE_GREY = "vision_transformer_image_grey" 
DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE = "deep_vit_grey_256_image"


class Deepfake_ECG_Dataset(torch.utils.data.Dataset):
    """
    Deepfake ECG dataset filtered to only include normal ECGs
    Contains 121977 ECGs
    ECG signals are returned as a 1D tensor (40k numbers)
    Parameters are returned as a 1D tensor
    """

    def __init__(self, parameter=None, output_type=DEFAULT_OUTPUT_TYPE):
        super(Deepfake_ECG_Dataset, self).__init__()
        self.output_type = output_type

        if parameter not in [HR_PARAMETER, QRS_PARAMETER, PR_PARAMETER, QT_PARAMETER]:
            raise ValueError("Invalid parameter")
        
        # NOTE : Uncomment below lines and comment out the next few lines

        # load the ground truth labels
        self.ground_truths = pandas.read_csv(
            "datasets/deepfake_ecg/filtered_all_normals_121977_ground_truth.csv"
        )

        # self.ground_truths = pandas.read_csv(
        #     "D:/SEM_07/FYP/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/code/datasets/deepfake_ecg/filtered_all_normals_121977_ground_truth.csv"
        # )

        if parameter == HR_PARAMETER:
            parameter = torch.tensor(
                self.ground_truths["avgrrinterval"].values, dtype=torch.float32
            )
            # calculate HR
            self.parameter = 60 * 1000 / parameter
        elif parameter == QRS_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["qrs"].values, dtype=torch.float32
            )
        elif parameter == PR_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["pr"].values, dtype=torch.float32
            )
        elif parameter == QT_PARAMETER:
            self.parameter = torch.tensor(
                self.ground_truths["qt"].values, dtype=torch.float32
            )

        # Dictionary to store loaded ASC files
        self.loaded_asc_files = {}

    def connect_ecgs_one_after_the_other(self, ecg_signals):
        # files have 8 columns, each column has one lead
        # extract each lead and flatten it
        # then append the flattened lead to allData so that allData has lead1, lead2, lead3, etc. one after the other
        allData = []
        for column in ecg_signals.columns:
            ecg_array = ecg_signals[column].values
            flattened_array = ecg_array.flatten()
            allData.extend(flattened_array)
        ecg_signals = torch.tensor(allData, dtype=torch.float32)
        ecg_signals = ecg_signals.reshape(-1)

        return ecg_signals

    def convert_to_DEFAULT_OUTPUT_TYPE(self, ecg_signals):
        ecg_signals = self.connect_ecgs_one_after_the_other(ecg_signals)
        ecg_signals = (ecg_signals + 3929.0) / 7642  # normalization : Range (0-1)
        
        # Transposing the ECG signals
        ecg_signals = ecg_signals.t()
        return ecg_signals

    def convert_to_DEFAULT_SPECTROGRAM_OUTPUT_TYPE(self, ecg_signals):
        ecg_signals = self.connect_ecgs_one_after_the_other(ecg_signals)
        _, _, Sxx = spectrogram(ecg_signals, 500)

        # change 0 values to 1 to avoid log(0) error
        Sxx[Sxx == 0] = 0.01

        Sxx = 10 * np.log10(Sxx)
        ecg_signals = Sxx.reshape(-1)
        return ecg_signals

    def convert_to_VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE(self, ecg_signals):
        # for now only use the first lead
        # TODO: use all leads

        # Set the desired width and height in pixels (required by vision transformer)
        width_pixels = 224
        height_pixels = 224

        # Calculate DPI based on the desired size
        dpi = 100

        # Calculate the figure size in inches
        fig_width = width_pixels / dpi
        fig_height = height_pixels / dpi

        # Create a plot with specified size
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.plot(ecg_signals[0])

        # Save the plot as a variable
        canvas = FigureCanvas(fig)
        canvas.draw()
        rgba_array = np.array(canvas.renderer.buffer_rgba())

        # Extract RGB values (discard alpha channel)
        # by default we have 4 channels. but we dont need the 4th one
        rgb_array = rgba_array[:, :, :3]

        ecg_signals = torch.tensor(rgb_array)
        # convert to float32 as requested by vision transformer
        ecg_signals = ecg_signals.to(torch.float32)

        # the tensor is in shape (224, 224, 3) but we need it in shape (3, 224, 224)
        ecg_signals = np.transpose(ecg_signals, (2, 1, 0))
        
        # close the figure. Otherwise higher CPU RAM usage
        plt.close(fig)

        return ecg_signals
    
    def convert_to_VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE_GREY(self, ecg_signals):
        ecg_signals_RGB = self.convert_to_VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE(ecg_signals)
    
        
        
        # by doing this, it will has 3 channels eventhough they are greyscale
        # can be set to 1 channel but that will not work with the ViT input
        # have to look more in to ViT input, if we need to send just one channel

        # NOTE : In ViT channels can be set to 1 for Grayscale images !
        ecg_signals_gray = F.rgb_to_grayscale(ecg_signals_RGB, num_output_channels=3) 
        return ecg_signals_gray
    

    def convert_to_DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE(self,ecg_signals):
        


    def __getitem__(self, index):
        filename = self.ground_truths["patid"].values[index]

        # Check if the ASC file is already loaded
        if filename in self.loaded_asc_files:
            ecg_signals = self.loaded_asc_files[filename]
        else:
            # Load the ASC file
             
            # NOTE : Uncomment below lines and comment out the next few lines

            ecg_signals = pandas.read_csv(
                f"datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/{filename}.asc",
                header=None,
                sep=" ",
            )

            # ecg_signals = pandas.read_csv(
            #     f"D:/SEM_07/FYP/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/code/datasets/deepfake_ecg/from_006_chck_2500_150k_filtered_all_normals_121977/{filename}.asc",
            #     header=None,
            #     sep=" ",
            # )
             
            if self.output_type == DEFAULT_OUTPUT_TYPE:
                ecg_signals = self.convert_to_DEFAULT_OUTPUT_TYPE(ecg_signals)
            elif self.output_type == DEFAULT_SPECTROGRAM_OUTPUT_TYPE:
                ecg_signals = self.convert_to_DEFAULT_SPECTROGRAM_OUTPUT_TYPE(
                    ecg_signals
                )
            elif self.output_type == VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE:
                ecg_signals = self.convert_to_VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE(
                    ecg_signals
                )
            elif self.output_type == VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE_GREY:
                ecg_signals = self.convert_to_VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE_GREY(
                    ecg_signals
                )
            elif self.output_type == DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE:
                ecg_signals = self.convert_to_DEEP_VIT_GREY_256_IMAGE_OUTPUT_TYPE(
                    ecg_signals
                )

            # Store the loaded ASC file in the dictionary
            self.loaded_asc_files[filename] = ecg_signals

        parameter = self.parameter[index].reshape(-1)

        return ecg_signals, parameter

    def __len__(self):
        if IS_FULL_DATASET:
            # run full dataset on servers
            return self.ground_truths.shape[0]

        # limit to 1000 on local computers
        # return 1000
        return 999
    
    
if __name__ == "__main__":
    from PIL import Image

    def tensor_to_image(tensor):
        # Assuming tensor is in shape (3, 224, 224) and in torch.float32 format
        
        # First, convert it to numpy and transpose it back to (224, 224, 3)
        # Make sure to clip values to the range [0, 255] and convert to uint8
        np_image = tensor.cpu().detach().numpy().transpose(1, 2, 0)
        np_image = np.clip(np_image, 0, 255).astype(np.uint8)
        
        # Now, convert numpy array to PIL Image
        image = Image.fromarray(np_image)
        
        return image
    
    def test_dataset_loading_and_processing_as_images_for_ViT():
        print("Main started, starting to get dataset as images for Vision Transformers")

        dataset = Deepfake_ECG_Dataset(
            parameter=HR_PARAMETER,
            output_type=VISION_TRANSFORMER_IMAGE_OUTPUT_TYPE_GREY,
        )
        print("Dataset loaded as images for Vison Transformers, getting the first item")

        first_item = dataset[0]
        data_tensor, second_tensor = first_item

        # Convert the tensor back to an image
        ecg_image = tensor_to_image(data_tensor)

        # Save the image to disk
        ecg_image.save("./datasets/deepfake_ecg/ecg_image_test_for_ViT_grey.png", "PNG")
        print("Image saved successfully.")
        
    #TODO : more tests can be added to test spectrogram and ...
    test_dataset_loading_and_processing_as_images_for_ViT()
        


    
    
    
