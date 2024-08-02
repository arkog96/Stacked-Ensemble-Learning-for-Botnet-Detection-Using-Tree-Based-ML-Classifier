This repository contains the complete codebase for the implementation of a project titled **["Stacked Ensemble Learning for Botnet Detection using Tree-based Machine Learning Classifier"]**, conducted at Dalhousie University. 

## Abstract of the Thesis
The increasing prevalence of botnet attacks poses significant threats to network security, necessitating advanced detection mechanisms. This paper presents a novel approach to botnet detection leveraging a tree-based machine learning algorithm for stacked ensemble learning. Our methodology employs Decision Trees, Random Forest, and AdaBoost as base classifiers in the level 0 layer. The performance of these classifiers is evaluated on validation data, and the best-performing model is subsequently utilized as the meta-classifier in the level 1 layer to generate the final prediction.  This stacked ensemble framework not only enhances detection accuracy but also offers robustness against diverse botnet attack vectors, providing a comprehensive solution for network intrusion detection systems.

## Dataset 
In this project, a publicly available dataset named, N-BaIoT, is utilized which is gathered using 9 commercial IoT devices. This dataset is created by gathering network traffic data originating from IoT devices that have been compromised by BASHLITE and Mirai botnets. For this project, the records gathered using last IoT device named "SimpleHome_XCS7_1003_WHT_Security_Camera" is utilized. For this work, the dataset is divided into 2 subjects for running two different types of classification tasks. The two subsets are:

* DS1: The first subset is categorized into two classes: Normal and Malicious. (Binary Classification - 2 Classes)
* DS2: The second subset is divided into three categories: Normal, Bashlite, and Mirai. (Muticlass Classification - 3 Classes)

## Data Undersampling


## Performance Evaluation 
The performance of the proposed ME-IDS framework is assessed against three benchmark ensemble methods, namely Stacked Ensemble, Concatenation Ensemble, and Confidence Averaging, as well as the three base classifiers used in the proposed weighted ensemble schemes within the framework.

| Method          | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------:|:----------:|:----------:|:----------:|:----------:|
| Stacked Ensemble| 85.85    | 87.47     | 84.75  | 85.29    |
| Concatenation Ensemble| 95.31 | 96.07     | 94.78  | 95.20    |
| Confidence Averaging   | 97.72 | 97.96     | 97.50  | 97.69    |
| VGG16-SA        | 98.72    | 98.79     | 98.63  | 98.70    |
| VGG16-TPE       | 99.15    | 99.17     | 99.11  | 99.14    |
| VGG16-RS        | 99.43    | 99.43     | 99.43  | 99.43    |
| ME-IDS          | 99.72    | 99.74     | 99.68  | 99.71    |

**<p align="center">Figure 4: Performance improvement of ME-IDS compared to base classifiers and three benchmark ensemble methods using selected features.</p>**
<p align="center">
<img src="https://github.com/arkog96/Weighted-Ensemble-Transfer-Learning-based-Intrusion-Detection-System-/blob/main/Figures/Performance%20Improvement.jpg" width="450" />
</p>

## System Requirements
 * Python 3.8
 * [Scikit-learn](https://scikit-learn.org/)
 * [Tensorflow 2.8.0](https://pypi.org/project/tensorflow/2.8.0/)
 * [Keras 2.8.0](https://pypi.org/project/keras/)
 * [Pillow (PIL)](https://pillow.readthedocs.io/)
 * [OpenCV](https://opencv.org/)
 * [Hyperopt](http://hyperopt.github.io/hyperopt/)

## Contact Information
In case of any enquiry, question or collaboration opportunities, kindly reach out to me at:
* Email: [arka.ghosh@dal.ca](mailto:arka.ghosh@dal.ca)
* LinkedIn: [Arka Ghosh](https://www.linkedin.com/in/llarkaghoshll/) 
