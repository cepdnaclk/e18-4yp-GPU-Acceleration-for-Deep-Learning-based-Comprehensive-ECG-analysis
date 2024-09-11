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

## Experiment Setup and Implementation
<p align="justify">
As the first step, multiple experiments were run to find out the most suitable model architecture to be used for the transfer learning task. For that different architectures were tested out on the Deepfake dataset as well as the PTB-XL dataset to figure out a model performing well on both regression as well as classification on ECG signals. </p>

<p align="justify">
After the model selection was done, all the following experiments were run using that model architecture. Prior to any transfer learning, baseline resutls were obtained for the PTB-XL five class classification task. Then the transfer learning approach was divided into two sections.  </p>

1. Transfer Learning within the same real dataset ( PTB-XL and PTB-XL PLUS)
2. Transfer Learning from synthetic dataset to real dataset ( Deepfake dataset and PTB-XL PLUS)

<p align="justify">
Transfer learning within the same dataset was tried out to test the feasibility of the methodology of transfer learning from regression to classification, and after that the main research question; using synthetic ECG signals to pretrain a ML model and then use that to classify diseases on a realdataset was tested.</p>

All the implementation was done using pytorch framework in python. For the first part, different model architectures, namely RNN, CNN, LSTM, VIT and 1D-CNN were tried out to find a suitable model architecture for the task. 

## Results and Analysis

<!--
A series of experiments were conducted to identify the optimal model for both regression and classification tasks on ECG signals. Different model architectures were evaluated: RNN, LSTM, CNN,  ViT, and 1D-CNN.

For regression, **1D-CNN** demonstrated superior performance, particularly with the lowest Mean Absolute Error (MAE) across multiple ECG parameters (HR, QRS, PR, QT). The performance of all models is summarized in Table 1.

#### Table 1: Regression Model Comparison
<div align="center">
    
| Model               | HR (Train MAE / Val MAE) | QRS (Train MAE / Val MAE) | PR (Train MAE / Val MAE) | QT (Train MAE / Val MAE) |
|---------------------|--------------------------|----------------------------|--------------------------|--------------------------|
| **RNN**             | 0.608 / 0.669            | 0.219 / 0.214              | -                        | -                        |
| **LSTM**            | 1.452 / 1.416            | 6.819 / 6.896              | 13.504 / 13.536          | 13.892                   |
| **CNN**             | 5.875 / 5.870            | 7.002 / 6.872              | -                        | -                        |
| **ViT**             | 5.975 / 5.869            | 7.979 / 7.930              | 23.819 / 24.012          | 17.320 / 16.817          |
| **1D-CNN**          | 1.237 / 0.706            | 3.259 / 3.007              | 5.801 / 5.110            | 6.351 / 4.130            |

</div>

For classification, the **1D-CNN** again outperformed other models, achieving the highest accuracy and AUC on both the training and validation sets (see Table 2).

#### Table 2: Classification Model Comparison

<div align="center">
    
| Model               | Train Accuracy / AUC | Val Accuracy / AUC |
|---------------------|----------------------|--------------------|
| **RNN**             | 0.559 / 0.758        | 0.557 / 0.736      |
| **LSTM**            | 0.560 / 0.779        | 0.551 / 0.770      |
| **CNN**             | 0.560 / 0.775        | 0.551 / 0.769      |
| **ViT**             | 0.560 / 0.776        | 0.551 / 0.769      |
| **1D-CNN**          | 0.795 / 0.905        | 0.776 / 0.902      |

</div>

### Transfer Learning within PTB-XL Dataset

Following the model selection, transfer learning was applied using the **1D-CNN** model to improve classification performance. Transfer learning was first applied between subsets of the PTB-XL dataset, where models were initially trained on ECG parameters (HR, QRS, PR, QT) and then transferred for classification tasks. The results are summarized in Table 3.

#### Table 3: Transfer Learning Results (PTB-XL)

<div align="center">
    
| Model               | Train Accuracy / AUC | Val Accuracy / AUC | Test Accuracy / AUC |
|---------------------|----------------------|--------------------|---------------------|
| **Baseline**        | 0.795 / 0.905        | 0.776 / 0.903      | 0.775 / 0.884       |
| **Transfer HR**     | 0.831 / 0.946        | 0.752 / 0.894      | 0.787 / 0.892       |
| **Transfer QRS**    | 0.802 / 0.915        | 0.791 / 0.892      | 0.785 / 0.906       |
| **Transfer PR**     | 0.812 / 0.924        | 0.785 / 0.902      | 0.766 / 0.883       |
| **Transfer QT**     | 0.838 / 0.949        | 0.808 / 0.901      | 0.783 / 0.893       |

</div>

### Transfer Learning from Deepfake Dataset to PTB-XL

In the final set of experiments, transfer learning was applied by pretraining the **1D-CNN** model on a synthetic "deepfake dataset" before transferring it to the PTB-XL dataset for classification tasks. The performance of these models is shown in Table 4.

#### Table 4: Transfer Learning Results (Deepfake to PTB-XL)

<div align="center">
    
| Model               | Train Accuracy / AUC | Val Accuracy / AUC | Test Accuracy / AUC |
|---------------------|----------------------|--------------------|---------------------|
| **Baseline**        | 0.795 / 0.905        | 0.776 / 0.903      | 0.775 / 0.884       |
| **Transfer HR**     | 0.801 / 0.904        | 0.785 / 0.906      | 0.764 / 0.887       |
| **Transfer QRS**    | 0.822 / 0.933        | 0.777 / 0.897      | 0.783 / 0.876       |
| **Transfer PR**     | 0.761 / 0.865        | 0.754 / 0.886      | 0.764 / 0.888       |
| **Transfer QT**     | 0.806 / 0.912        | 0.765 / 0.913      | 0.778 / 0.895       |

</div>

-->

## Conclusion 

## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository" 

 1. [Semester 7 report](./) 
 2. [Semester 7 slides](./) 
 3. [Semester 8 report](./) 
 4. [Semester 8 slides](./)
 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). 


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository](https://github.com/cepdnaclk/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis)
- [Project Page](https://cepdnaclk.github.io/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/)
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
