# NASA Turbofan Failure Prediction

This data analytics / machine learning project investigates the relationship between behavioural variables and the onset of failure (in terms of remaining engine cycles) for simulated operational turbofan data from a NASA research project. 

The project starts with an exploration of the datasets, folowed by the development of predictive models for engine Remaining Useful Life (RUL) based on current engine readings. Modelling techniques include linear regression and neural networks (using TF-Keras).

Training data is sourced from the NASA Prognostics Center Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan

An article about this investigation can be found at: https://www.cloudforesttechnologies.com/post/predictive-analytics-implementation-strategies

# Project Objectives

1. To analyse relationships between engine behaviour and remaining useful lifetime.
2. To develop predictive models for remaining useful lifetime.

# Exploring the Dataset

Datasets where explored using Jupyter Notebooks, with intitial checks for data quality followed by investigations for variable relationships. The quality of the data was generally very good, with very few instances of missing data or incorrect datatypes.

Strong linear correlations between a number of variables can be seen in the dataset, providing a strong foundation for subsetting variables for the predictive model:

![Image of Correlations](https://github.com/PMetcalf/nasa-turbofan-failure-prediction/blob/master/Reports/Figures/PM_Dataset_Linear_Correlations_2021_02_09-11_51_00.png)



Consult the Notebooks section of the repository for further information.

# Predictive Modelling

xxx

# Installation & Setup

The following packages are required to support this project:

numpy, pandas, matplotlib, seaborn, sklearn, keras.

# Clone

Clone this repository from: https://github.com/PMetcalf/nasa_turbofan_failure_prediction.git

# Acknowledgements

This project drew inspiration from work by Ali-Alhamaly (https://medium.com/@hamalyas_/jet-engine-remaining-useful-life-rul-prediction-a8989d52f194) and Roshan Alwis & Srinath Perea (https://www.infoq.com/articles/machine-learning-techniques-predictive-maintenance/)

Data was made available from the NASA Prognostics Center of Excellence.
