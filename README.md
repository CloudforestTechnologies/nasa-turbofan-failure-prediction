# NASA Turbofan Failure Prediction

This project builds a simple data pipeline and a series of machine learning models to predict the onset of failure in terms of remaining engine cycles for simulated turbofan data from NASA. 

Models from sklearn, TensorFlow (Keras) and PyTorch are trained on equivalent data sets and compared for performance and accuracy.

Training data is sourced from the NASA Prognostics Center Data Repository: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/#turbofan

# Uses

1. Engine data importation and visualisation.
2. Data cleaning and preprocessing techniques.
3. ML model training and optimisation.
4. Model evaluation and comparison.

# Installation & Setup

The following packages are required to support this project:

numpy, pandas, matplotlib, seaborn, sklearn, keras, torch. 

# Clone

Clone this repository from: https://github.com/PMetcalf/nasa_turbofan_failure_prediction.git

# Acknowledgements

This project drew inspiration from work by Ali-Alhamaly (https://medium.com/@hamalyas_/jet-engine-remaining-useful-life-rul-prediction-a8989d52f194) and Roshan Alwis & Srinath Perea (https://www.infoq.com/articles/machine-learning-techniques-predictive-maintenance/)

Data was made available from the NASA Prognostics Center of Excellence.

Remaining Useful Life Estimation: https://www.mathworks.com/help/predmaint/examples/similarity-based-remaining-useful-life-estimation.html#SimilarityBasedRULExample-10
