# HousingCostPredictor
Python ML models to predict housing costs


Purpose: The purpose of this lab was to delve into the world of supervised machine learning. With the real estate market being a popular subject, predicting housing costs is an apt topic to apply to machine learning models. 

Linear regression formula to predict the cost h_(θ)(x):
Univariate - h_(θ)(x)=θ_0 +θ_1x
Multivariate - h_θ(x)=θ_0 + θ_1(x)_1 + ... + θ_n(x)_n 
​

Data Clean-up and Prep:
- Dropped the "address" column 
- The split for the training and test size was 90%/10%, respectively
- Normalize data using the MinMax scaler via scikit-learn so that each feature will be a value between 0 and 1


Defining loss functions:
- L2 square loss function
- L1 absolute error function
- Pseudo Huber loss function

Comparison of theta0 and theta1 for each loss:
- an interactive panel that calculates the L1, L2, and Huber errors between the predicted values and actual data
- plots the regression line and data points to compare the errors and values of theta

Univariate Gradient Descent:
- Compute theta coefficients using Gradient descent ( L2 and Huber )

Extending model to Bivaraite
- Now predicting housing costs using avg. area house (x_1) and avg. area number of rooms (x_2)
  
