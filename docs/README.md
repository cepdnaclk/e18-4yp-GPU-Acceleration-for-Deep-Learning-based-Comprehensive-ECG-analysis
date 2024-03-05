---
layout: home
permalink: index.html

# Please update this with your repository name and title
repository-name: eYY-4yp-project-template
title:
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
<!--
6. [Experiment Setup and Implementation](#experiment-setup-and-implementation)
7. [Results and Analysis](#results-and-analysis)
8. [Conclusion](#conclusion)
9. [Publications](#publications)
10. [Links](#links) -->

---

<!-- 
DELETE THIS SAMPLE before publishing to GitHub Pages !!!
This is a sample image, to show how to add images to your page. To learn more options, please refer [this](https://projects.ce.pdn.ac.lk/docs/faq/how-to-add-an-image/)
![Sample Image](./images/sample.png) 
-->


## Abstract

The leading cause of death in humans, Cardiovascular diseases could be diagnosed by analysis of electrocardiogram (ECG) which is a non-invasive method that records electrical activity of the cardiac cycle. Due to the rise of privacy issues in using ECG records of patients for research purposes synthetic generated data with similar information and distribution have become an alternative. Attention based mechanism which is the basis of Transformer Neural Networks combined with other models such as Convolutional Neural Networks, Recurrent Neural Networks and Long-Short Term Memory have been used in ECG classification tasks using real patient ECG data with promising outcomes. But analysis of properties of ECG signals using attention based regression methods on synthetic data and transferring the learned parameters for fine tuning on limited real data is the interest area of this project.


## Related works

The intersection of machine learning (ML) techniques with electrocardiogram (ECG) data analysis has emerged as a promising area in healthcare technology, offering novel approaches to cardiac care. The application of ML for ECG regression and classification tasks has been identified as a transformative advancement, with the potential to enhance diagnostic accuracy and patient outcomes significantly.

Electrocardiograms, which graphically represent the heart's electrical activity, are crucial for diagnosing various cardiac conditions. Traditional analysis of ECGs has relied heavily on the expertise of clinicians, a process that can be subjective and time-consuming. The advent of ML in ECG analysis promises a shift towards more objective, efficient, and accurate interpretations. ML models, especially deep learning architectures like convolutional neural networks (CNNs), recurrent neural networks (RNNs), and Vision Transformers (ViT), have demonstrated remarkable capabilities in extracting meaningful patterns from ECG signals. These models have been successfully applied to predict critical cardiac parameters such as heart rate, QT interval, QRS duration, and ST segment changes directly from ECG waveforms. Accurate predictions of these parameters are essential for diagnosing cardiac conditions, assessing cardiovascular risk, and guiding therapeutic interventions.

The literature also highlights various challenges in ML-based ECG analysis, including issues related to data quality, dataset heterogeneity, and the interpretability of model outcomes. Despite these challenges, the integration of ML with ECG data has seen significant applications in clinical settings, such as the early detection of arrhythmias and personalized risk stratification. These advancements underscore the potential of ML to revolutionize cardiac care by enhancing the precision of diagnoses and the customization of treatment plans.

Our project builds upon these foundational insights, aiming to address some of the identified challenges and explore new avenues in ECG analysis through advanced ML models. By focusing on the integration of state-of-the-art ML techniques with high-quality ECG datasets, our work seeks to further the capabilities of automated ECG interpretation, contributing to the ongoing evolution of cardiac healthcare technologies.

## Methodology

<div style="text-align: center;">
    <img src="https://machinelearningmastery.com/wp-content/uploads/2021/08/attention_research_1.png" width="330" height="415" alt="Sample Image" />
</div>
<br>
In the evolving landscape of electrocardiogram (ECG) analysis, the integration of advanced machine learning techniques, particularly the adoption of transformer models like Vision Transformers (ViT), represents a promising frontier. Our project is at the forefront of exploring the use of Vision Transformers for the nuanced task of ECG signal analysis. The inherent capability of transformer models to handle sequential data, despite being initially designed for vision-based tasks, offers a novel approach to interpreting the complex patterns embedded in ECG waveforms. This methodological pivot underscores our commitment to leveraging cutting-edge technology to enhance diagnostic accuracy and efficiency in cardiology.

Currently, our research is focused on harnessing Vision Transformers to analyze ECG signals with the goal of predicting vital cardiac parameters accurately. These parameters include, but are not limited to, heart rate, QT interval, QRS duration, and ST segment changes. Accurate prediction of these parameters is crucial for diagnosing various cardiac conditions, assessing cardiovascular risk, and guiding treatment decisions effectively. Preliminary experiments utilizing Vision Transformers have shown promising results, indicating the potential of these models to learn and interpret the intricate patterns in ECG data effectively.

Initially transformer model will be trained on the Deepfake ECG dataset to predict values and using that predictive capabilities of Vision Transformers in ECG analysis, our project aims to extend this methodology to perform classification tasks on the PTB-XL dataset, a comprehensive ECG dataset that is widely used for benchmarking in the field. The strategy involves employing the trained Vision Transformer models as feature extractors or initial layers in a larger machine learning pipeline. By doing so, we intend to leverage the nuanced understanding of ECG signals these models have developed to classify various cardiac conditions accurately. This approach not only capitalizes on the strengths of Vision Transformers in handling sequential and spatial data but also introduces a transfer learning aspect to our methodology. Transfer learning will allow us to adapt models pretrained on predicting ECG parameters to the task of classifying different classes within the PTB-XL dataset, potentially reducing the need for extensive retraining and enabling more efficient model adaptation.
<!--
## Experiment Setup and Implementation

## Results and Analysis

## Conclusion -->

<!--## Publications
[//]: # "Note: Uncomment each once you uploaded the files to the repository" -->

<!-- 1. [Semester 7 report](./) -->
<!-- 2. [Semester 7 slides](./) -->
<!-- 3. [Semester 8 report](./) -->
<!-- 4. [Semester 8 slides](./) -->
<!-- 5. Author 1, Author 2 and Author 3 "Research paper title" (2021). [PDF](./). -->


## Links

[//]: # ( NOTE: EDIT THIS LINKS WITH YOUR REPO DETAILS )

- [Project Repository]([https://github.com/cepdnaclk/repository-name](https://github.com/cepdnaclk/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis))
- [Project Page]([https://cepdnaclk.github.io/repository-name](https://cepdnaclk.github.io/e18-4yp-GPU-Acceleration-for-Deep-Learning-based-Comprehensive-ECG-analysis/))
- [Department of Computer Engineering](http://www.ce.pdn.ac.lk/)
- [University of Peradeniya](https://eng.pdn.ac.lk/)

[//]: # "Please refer this to learn more about Markdown syntax"
[//]: # "https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet"
