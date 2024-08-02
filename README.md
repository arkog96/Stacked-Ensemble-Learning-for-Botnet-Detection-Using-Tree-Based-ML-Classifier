This repository contains the complete codebase for the implementation of a project titled **"Stacked Ensemble Learning for Botnet Detection using Tree-based Machine Learning Classifier"**, conducted at Dalhousie University. 

## Abstract of the Thesis
The increasing prevalence of botnet attacks poses significant threats to network security, necessitating advanced detection mechanisms. This paper presents a novel approach to botnet detection leveraging a tree-based machine learning algorithm for stacked ensemble learning. Our methodology employs Decision Trees, Random Forest, and AdaBoost as base classifiers in the level 0 layer. The performance of these classifiers is evaluated on validation data, and the best-performing model is subsequently utilized as the meta-classifier in the level 1 layer to generate the final prediction.  This stacked ensemble framework not only enhances detection accuracy but also offers robustness against diverse botnet attack vectors, providing a comprehensive solution for network intrusion detection systems.

## Dataset 
In this project, a publicly available dataset named, N-BaIoT, is utilized which is gathered using 9 commercial IoT devices. This dataset is created by gathering network traffic data originating from IoT devices that have been compromised by BASHLITE and Mirai botnets. For this project, the records gathered using last IoT device named "SimpleHome_XCS7_1003_WHT_Security_Camera" is utilized. For this work, the dataset is divided into 2 subjects for running two different types of classification tasks. The two subsets are:

* DS1: The first subset is categorized into two classes: Normal and Malicious. (Binary Classification - 2 Classes)
* DS2: The second subset is divided into three categories: Normal, Bashlite, and Mirai. (Muticlass Classification - 3 Classes)

## Data Undersampling
The N-BaIoT dataset is highly imbalanced, as shown in Figure 1 with bar and pie charts for both DS1 and DS2. In DS1, only 2.3% of the data belongs to the normal class, while 97.7% is attack data. Similarly, DS2 has significantly fewer instances of the normal class compared to the other two classes. This imbalance biases ML classifiers towards the majority class, necessitating data balancing before training.

**<p align="center">Figure 1: Data distribution of DS1 & DS2 before undersampling.</p>**
<p align="center">
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS1%20Bar%20Chart.png" width="400" />
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS2%20Bar%20Chart.png" width="400" />
</p>
<p align="center">
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS1%20Pie%20Chart.png" width="400" />
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS2%20Pie%20Chart.png" width="400" />
</p>

This imbalance can result in low sensitivity (high false negatives) and overestimation of the model's accuracy. To address this problem, undersampling is employed, which involves reducing the number of records in the majority class to match the quantity of the minority class. After undersampling, the data distribution of both DS1 and DS2 is shown in Figure 2. 

**<p align="center">Figure 2: Data distribution of DS1 & DS2 after undersampling.</p>**
<p align="center">
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS1_Undersampled%20Bar.png" width="400" />
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS2_Undersampled%20Bar.png" width="400" />
</p>
<p align="center">
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS1_Undersampled%20Pie.png" width="400" />
<img src="https://github.com/arkog96/Stacked-Ensemble-Learning-for-Botnet-Detection-Using-Tree-Based-ML-Classifier/blob/main/Figures/DS2_Undersampled%20Pie.png" width="400" />
</p>

## Performance Evaluation 
The result of the base classifiers and stacked ensemble is as follows, for both DS1 and DS2:

**<u>DS1</u>**
| Classifier          | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------:|:----------:|:----------:|:----------:|:----------:|
| Decision Tree| 99.9658   | 99.9659     | 99.9658  | 99.9658     |
| Random Forest| 99.9487 | 99.9488   | 99.9487  | 99.9488     |
| AdaBoost   | 99.9488 | 99.9488     | 99.9487  | 99.9487   |
| Stacked Ensemble        | 99.9659    | 99.9659     | 99.9659  | 99.9659    |

**<u>DS2</u>**
| Classifier          | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) |
|-----------------:|:----------:|:----------:|:----------:|:----------:|
| Decision Tree| 99.9203   | 99.9204     | 99.9203  | 99.9203    |
| Random Forest| 99.9829| 99.9829   | 99.9487  | 99.9488     |
| AdaBoost   | 99.9488 | 99.9488     | 99.9829  | 99.9829  |
| Stacked Ensemble        | 99.9829    | 99.9829     | 99.9829  | 99.9829    |



## System Requirements
 * Python 3.8
 * [Pandas](https://pandas.pydata.org/)
 * [NumPy](https://numpy.org/devdocs/)
 * [Matplotlib](https://matplotlib.org/)
 * [Scikit-learn](https://scikit-learn.org/)

## Contact Information
In case of any enquiry, question or collaboration opportunities, kindly reach out to me at:
* Email: [arka.ghosh@dal.ca](mailto:arka.ghosh@dal.ca)
* LinkedIn: [Arka Ghosh](https://www.linkedin.com/in/llarkaghoshll/) 
