---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis
title: GPU Acceleration for Deep-Learning based Comprehensive ECG analysis
---

[comment]: # "This is the standard layout for the project, but you can clean this and use your own template"

# GPU Acceleration for Deep-Learning based Comprehensive ECG analysis

#### Team

- E/18/098, Ishan Fernando, [email](mailto:e18098@eng.pdn.ac.lk)
- E/18/100, Adeepa Fernando, [email](mailto:e18100@eng.pdn.ac.lk)
- E/18/155, Ridma Jayasundara, [email](mailto:e18155@eng.pdn.ac.lk)

#### Supervisors

- Prof. Roshan G. Ragel, [email](mailto:roshanr@eng.pdn.ac.lk)
- Assoc. Prof. Vajira Thambawita, [email](mailto:vajira@simula.no)
- Dr. Isuru Nawinne, [email](mailto:isurunawinne@eng.pdn.ac.lk)

#### Table of content

1. [Abstract](#abstract)
2. [Related works](#related-works)
3. [Methodology](#methodology)
4. [Links](#links)
5. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
6. [Results and Analysis](#results-and-analysis)
7. [Conclusion](#conclusion)
8. [Publications](#publications)
9. [Links](#links) 

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract

<p align="justify">
The leading cause of death in humans, Cardiovascular diseases could be diagnosed by analysis of electrocardiogram (ECG) which is a non-invasive method that records electrical activity of the cardiac cycle. Due to the rise of privacy issues in using ECG records of patients for research purposes, synthetically generated data with similar information and distribution have become an alternative. We explore the possibility of using synthetic ECG data with regression labels to improve the accuracy of classification of diseases based on real ECG signals.</p>

## Related works

<p align="justify">
The intersection of machine learning (ML) techniques with electrocardiogram (ECG) data analysis has emerged as a promising area in healthcare technology, offering novel approaches to cardiac care. The application of ML for ECG regression and classification tasks has been identified as a transformative advancement, with the potential to enhance diagnostic accuracy and patient outcomes significantly. </p>

<p align="justify">
Electrocardiograms, which graphically represent the heart's electrical activity, are crucial for diagnosing various cardiac conditions. Traditional analysis of ECGs has relied heavily on the expertise of clinicians, a process that can be subjective and time-consuming. The advent of ML in ECG analysis promises a shift towards more objective, efficient, and accurate interpretations. ML models, especially deep learning architectures like convolutional neural networks (CNNs), recurrent neural networks (RNNs), and Vision Transformers (ViT), have demonstrated remarkable capabilities in extracting meaningful patterns from ECG signals. These models have been successfully applied to predict critical cardiac parameters such as heart rate, QT interval, QRS duration, and ST segment changes directly from ECG waveforms. Accurate predictions of these parameters are essential for diagnosing cardiac conditions, assessing cardiovascular risk, and guiding therapeutic interventions. </p>

<p align="justify">
The literature also highlights various challenges in ML-based ECG analysis, including issues related to data quality, dataset heterogeneity, and the interpretability of model outcomes. Despite these challenges, the integration of ML with ECG data has seen significant applications in clinical settings, such as the early detection of arrhythmias and personalized risk stratification. These advancements underscore the potential of ML to revolutionize cardiac care by enhancing the precision of diagnoses and the customization of treatment plans. </p>

<p align="justify">
Our project builds upon these foundational insights, aiming to address some of the identified challenges and explore new avenues in ECG analysis through advanced ML models. By focusing on the integration of state-of-the-art ML techniques with high-quality ECG datasets, our work seeks to further the capabilities of automated ECG interpretation, contributing to the ongoing evolution of cardiac healthcare technologies. </p>


## Methodology
The study employed two primary datasets: the PTB-XL dataset, containing 21,837 real 12-lead ECG records from 18,885 patients, and a DeepFake ECG dataset of synthetically generated signals. The PTB-XL dataset was preprocessed and divided into two subsets (PTB-XL-A and PTB-XL-B) for transfer learning experiments.
For both classification and regression tasks, a 1D Convolutional Neural Network (1D-CNN) model was used after comparing performance with other architectures. The classification task involved categorizing ECG signals into five diagnostic categories, while the regression task focused on predicting four key ECG parameters: Heart Rate (HR), PR interval, QT interval, and QRS complex.
Data preprocessing included normalization and splitting into train (70%), validation (10%), and test (20%) sets. Data augmentation techniques were applied to enhance model robustness. The models were trained using the Adam optimizer with a learning rate scheduler. For classification, a categorical cross-entropy loss function was used, while Mean Absolute Error (MAE) was employed for regression.
Performance evaluation used Area Under the Curve (AUC) for classification tasks and MAE for regression tasks. Hyperparameter tuning and regularization techniques were applied to optimize model performance and prevent overfitting.
The transfer learning process involved two approaches:

1. Combining all four regression models into a single classification model 
2. Using only the QRS regression model for transfer learning

These approaches were first tested within the PTB-XL dataset subsets, then extended to transfer learning from the DeepFake dataset to the PTB-XL dataset. This process aimed to leverage knowledge gained from regression tasks to improve classification performance and to test the effectiveness of using synthetic data for training models applicable to real-world scenarios.
The implementation utilized the PyTorch library and leveraged GPU acceleration to expedite the training process, allowing for quicker iterations and more extensive hyperparameter searches.


## Experiment Setup and Implementation

The **PTB-XL** dataset, curated by the PhysioNet community, is a large and widely recognized resource for electrocardiogram (ECG) analysis. It contains 21,837 clinical ECG records collected from 18,885 patients. Each record is a 10-second, 12-lead ECG sampled at 500 Hz. The dataset is accompanied by extensive metadata, including patient demographics and diagnostic labels, which fall into five primary categories:

- **NORM** (Normal ECG)
- **CD** (Myocardial Infarction)
- **STTC** (ST/T Changes)
- **MI** (Conduction Disturbance)
- **HYP** (Hypertrophy)

For the experiments in this project, 8 of the 12 leads were used to ensure compatibility with synthetic ECG data in the transfer learning process. These leads include: Lead I, Lead II, V1, V2, V3, V4, V5, and V6.

The PTB-XL dataset was preprocessed to remove noise and clean the data. Only ECG signals with a single class label (from the 5-class set) were used for classification. The dataset was then divided into two subsets, **PTB-XL-A** and **PTB-XL-B**, ensuring balanced representation of the five diagnostic categories in both sets.
<div align="center">
<img width="1678" alt="data_dis_" src="https://github.com/user-attachments/assets/a5fff096-bca9-4529-b91c-32b1703db713">
</div>

<p align="justify">
As the first step, multiple experiments were run to find out the most suitable model architecture to be used for the transfer learning task. For that different architectures were tested out on the Deepfake dataset as well as the PTB-XL dataset to figure out a model performing well on both regression as well as classification on ECG signals. </p>

The model architecture of the selected 1D-CNN is as below.
<div align="center">
<img width="805" alt="model_archi" src="https://github.com/user-attachments/assets/16515688-467c-46d2-bbad-3f2ad7b3115b">
</div>

<p align="justify">
After the model selection was done, all the following experiments were run using that model architecture. Prior to any transfer learning, baseline resutls were obtained for the PTB-XL five class classification task. Then the transfer learning approach was divided into two sections.  </p>

1. Transfer Learning within the same real dataset ( PTB-XL and PTB-XL PLUS)
2. Transfer Learning from synthetic dataset to real dataset ( Deepfake dataset and PTB-XL PLUS)

<p align="justify">
Transfer learning within the same dataset was tried out to test the feasibility of the methodology of transfer learning from regression to classification, and after that the main research question; using synthetic ECG signals to pretrain a ML model and then use that to classify diseases on a realdataset was tested.</p>

All the implementation was done using pytorch framework in python. For the first part, different model architectures, namely RNN, CNN, LSTM, VIT and 1D-CNN were tried out to find a suitable model architecture for the task. 

When the models were trained to predict values such as HR, QRS, PR and QT and then to classify the diseases, the model layout has to be changed in order to support the classification task. Image down below shows the changes done in the model architecture to support classification and regression tasks.
<div align="center">
<img width="535" alt="Screenshot 2024-09-12 at 12 19 35 AM" src="https://github.com/user-attachments/assets/772fc264-3fdd-4954-a54f-e9e7262d7e42">
</div>

## Results and Analysis


A series of experiments were conducted to identify the optimal model for both regression and classification tasks on ECG signals. Different model architectures were evaluated: RNN, LSTM, CNN,  ViT, and 1D-CNN.

For regression, **1D-CNN** demonstrated superior performance, particularly with the lowest Mean Absolute Error (MAE) across multiple ECG parameters (HR, QRS, PR, QT). The performance of all models is summarized in Table 1.

##### Table 1: Regression Model Comparison
<div align="center">
    
| Model               | HR (Train MAE / Val MAE) | QRS (Train MAE / Val MAE) | PR (Train MAE / Val MAE) | QT (Train MAE / Val MAE) |
|---------------------|--------------------------|----------------------------|--------------------------|--------------------------|
| **RNN**             | 5.569 / 5.530            | 6.983 / 6.998              | 13.489 / 13.403          | 16.243 / 16.494          |
| **LSTM**            | 5.536 / 5.546            | 6.819 / 6.896              | 13.504 / 13.536          | 13.892 / 13.892          |
| **CNN**             | 5.875 / 5.870            | 7.002 / 6.872              | 13.496 / 13.418          | 16.239 / 16.378          |
| **ViT**             | 5.975 / 5.869            | 7.979 / 7.930              | 23.819 / 24.012          | 17.320 / 16.817          |
| **1D-CNN**          | 1.237 / 0.706            | 3.259 / 3.007              | 5.801 / 5.110            | 6.351 / 4.130            |

</div>

For classification, the **1D-CNN** again outperformed other models, achieving the highest accuracy and AUC on both the training and validation sets (see Table 2).

##### Table 2: Classification Model Comparison

<div align="center">
    
| Model               | Train Accuracy / AUC | Val Accuracy / AUC |
|---------------------|----------------------|--------------------|
| **RNN**             | 0.559 / 0.758        | 0.557 / 0.736      |
| **LSTM**            | 0.560 / 0.779        | 0.551 / 0.770      |
| **CNN**             | 0.560 / 0.775        | 0.551 / 0.769      |
| **ViT**             | 0.560 / 0.776        | 0.551 / 0.769      |
| **1D-CNN**          | 0.795 / 0.905        | 0.776 / 0.902      |

</div>

## Conclusion 
This study explores the use of deep learning, particularly 1D Convolutional Neural Networks (1D-CNN), for ECG signal analysis, focusing on both regression tasks (predicting ECG parameters) and classification tasks (diagnosing cardiac conditions). The research utilizes the PTB-XL dataset for real ECG data and a DeepFake ECG dataset for synthetic data, implementing transfer learning techniques within the PTB-XL dataset and from synthetic to real data. Key findings include the superior performance of 1D-CNN models over other architectures, improved classification performance through transfer learning within the PTB-XL dataset, and mixed but promising results when transferring from synthetic to real data. The study contributes to the development of more accurate and efficient ECG analysis tools, addressing data privacy concerns in medical research and potentially improving cardiac disease diagnosis and patient care.  


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis)
- [Project Page](https://cepdnaclk.github.io/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
