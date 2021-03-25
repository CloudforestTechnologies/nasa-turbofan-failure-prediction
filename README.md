# NASA Turbofan Failure Prediction

This data analytics / machine learning project investigates the relationship between sensor data and the onset of failure (in terms of remaining engine cycles) for simulated  turbofan engines from a NASA research project. 

The project starts with an exploration of the datasets, folowed by the development of predictive models for engine Remaining Useful Life (RUL) based on current engine readings. Modelling techniques include linear regression and neural networks (using TF-Keras).

![Image of Turbofan](https://github.com/PMetcalf/nasa-turbofan-failure-prediction/blob/master/Miscellaneous/208009339-huge.jpg)

Training data is sourced from the NASA Prognostics Center Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan

An article about this investigation can be found at: https://www.cloudforesttechnologies.com/post/predictive-analytics-implementation-strategies

# Project Objectives

1. To analyse relationships between engine behaviour and remaining useful lifetime.
2. To develop predictive models for remaining useful lifetime.

# Exploring the Dataset

Datasets where explored using Jupyter Notebooks, with intitial checks for data quality followed by visualisation and exploration of variable relationships. The quality of the data was generally very good, with very few instances of missing data or incorrect datatypes, although accompanying documentation indicates the presence of noise on some sensors.

Strong linear correlations between a number of variables can be seen in the dataset, providing a strong foundation for subsetting variables for the predictive model:

![Image of Correlations](https://github.com/PMetcalf/nasa-turbofan-failure-prediction/blob/master/Reports/Figures/PM_Dataset_Linear_Correlations_2021_02_09-11_51_00.png)

And a number of variables show clear trends as a point of failure is approached, providing the starting point for predictive modelling:

![Image of Time Series](https://github.com/PMetcalf/nasa-turbofan-failure-prediction/blob/master/Reports/Figures/PM_Time_Series_All_Engines_2021_02_09-11_49_28_v2.png)

Consult the Notebooks section of the repository for further information.

# Predictive Modelling

Having analysed some of the relationships between sensor data and remaining useful life, a series of machine learning models were developed to predict how many cycles an engine had before a failure event based on realtime readings. 

A number of different types of machine learning model were investigated, including linear and polynomial regression, decision tree regressors, random forest regressors and neural network regressors (using TensorFlow and Keras). The dataset was preprocessed to select the most influential input variables, and remaining cycles to failure was used as the target variable. 

An out-of-the-box linear regression model trained on the data achieved RMSEs and MAEs of 46.4 and 35.7 cycles respectively - The average remaining number of cycles for each engine is around 270 cycles, so this works out as a rule-of-thumb accuracy of around 15%. Neural networks performed at a similar level.

Visualising predicted versus actual life cycles shows that the predictive models become more accurate as failure is approached.

![Image of Time Series](https://github.com/PMetcalf/nasa-turbofan-failure-prediction/blob/master/Reports/Figures/PM_Time_Series_All_Engines_2021_02_09-11_49_28.png)

Consult the Notebooks section of the repository for further information.

# Installation & Setup

The following packages are required to support this project:

numpy, pandas, matplotlib, seaborn, sklearn, keras.

# Clone

Clone this repository from: https://github.com/PMetcalf/nasa_turbofan_failure_prediction.git

# Acknowledgements

This project drew inspiration from work by Ali-Alhamaly (https://medium.com/@hamalyas_/jet-engine-remaining-useful-life-rul-prediction-a8989d52f194) and Roshan Alwis & Srinath Perea (https://www.infoq.com/articles/machine-learning-techniques-predictive-maintenance/)

Data was made available from the NASA Prognostics Center of Excellence.
