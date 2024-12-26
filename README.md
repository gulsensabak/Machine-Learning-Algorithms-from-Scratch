Machine Learning Project: Coffee Data Analysis
==============================================

Overview
--------
This project implements various machine learning algorithms from scratch and compares them with Scikit-learn implementations. 
The goal is to analyze and classify coffee data (Arabica vs Robusta) based on physical attributes. 
The project includes tasks such as similarity analysis, feature extraction, clustering, and classification.

Project Structure
-----------------
1. ClassSimilarity/
   - Contains scripts and notebooks for calculating intra-class and inter-class similarity metrics 
     (e.g., Euclidean, Manhattan, Cosine).
   - Outputs include detailed metrics for analyzing the separability of Arabica and Robusta classes.

2. data/
   - Contains data synthesizing codes required for completing the dataset.
   - Already done, you do not need to run them.

3. featureExtraction/
   - Scripts for extracting relevant features (e.g., width, height, depth, weight) from the dataset.
   - Implements feature selection techniques to improve model performance.

4. kMeansClass/
   - Contains the implementation of K-means clustering from scratch.
   - Includes visualizations (PCA-reduced plots) of clustering results and analysis of cluster quality.

5. KNN/
   - Scripts for implementing the k-Nearest Neighbors (KNN) algorithm from scratch.
   - Includes performance evaluation and comparison with Scikit-learn’s KNN implementation.

6. LogisticRegression/
   - Implementation of logistic regression from scratch, with support for L2 regularization.
   - Tasks include:
     - Monitoring training for overfitting.
     - Comparison of runtime and accuracy with Scikit-learn’s LogisticRegression.

7. randomForest/
   - Implementation of a Random Forest classifier.
   - Includes performance evaluation.


8. supportVector/
   - Scripts for implementing Support Vector Machines (SVM) from scratch.
   - Comparison of performance with Scikit-learn’s SVM implementation.

9. svm_vs_logistic/
   - Side-by-side comparison of SVM and logistic regression models.
   - Evaluates accuracy, runtime, and decision boundaries for both models.
   
10. analysis/
   - Comparison of all trained models with metrics accuracy, precision, recall, F1 score, AUROC, and run time. 

10. coffeeDataSynthesized.xlsx
    - The main dataset containing attributes for Arabica and Robusta coffee samples.
    - Attributes include width, height, depth, weight, country, origin, altitude, variety, process, flavor, acidity,and type (target variable).

How to Run
----------
1. Set up Environment:
   - Ensure Python 3.x is installed.
   - Install the required libraries:
     pip install numpy pandas matplotlib scikit-learn xgboost scipy seaborn cvxopt

2. Explore Specific Tasks:
   - Navigate to the folder corresponding to the task (e.g., `LogisticRegression/`).
   - Run the Jupyter notebooks inside the folder.
   - Results (metrics, plots) will be saved in the corresponding files.




