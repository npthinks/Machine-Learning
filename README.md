# Machine-Learning
Machine Learning Project Report

**Description**
The machine learning challenge aims to predict players’ responses to the question related to wellbeing based on a dataset from the Research Edition of the video game Power Wash Simulator.
The evaluation metric for the model is Mean Absolute Error (MAE). Firstly, extensive exploratory data analysis (EDA) was conducted to gain a deeper understanding of the dataset. This involved examining data types, descriptions, missing values, and outliers. Additionally, using the Matplotlib package, analyzed data distributions and correlations between numerical features. The EDA provided a foundation for to proceed with all aspects of modeling.
Furthermore, acquired domain knowledge by tracking down the most likely potential source of the dataset. The information gained related to the video game’s structure and the nature of the data collection also helped to make informed decisions during modeling.

**Feature Engineering**
The thorough exploratory data analysis (EDA) provided with a comprehensive understanding of the dataset, supplemented by domain knowledge from several articles about the game. During feature engineering, examined each feature meticulously and determined the best methods to handle both categorical and numerical variables.
For instance, applied dummy coding to the features ‘UserID’, ‘CurrentGameMode’, ‘QuestionTiming’, ‘CurrentTask’, and ‘LastTaskCompleted’. Initially clustered ‘TimeUtc’ into morning, afternoon, and night time frames, although this approach later proved ineffective. The ‘QuestionType’ feature was dropped as it contained only one categorical value, which would not contribute to the model's predictive power. For ‘CurrentSessionLength’, we used the StandardScaler to ensure that this feature contributed equally to the learning process of the model. Last but not least, transformed ‘TimeUtc’ by splitting the feature into ‘Year’, ‘Month’, ‘Day’, ‘Hour’, and ‘Minute’.
To address missing values (NaN), two strategies were adopted: (1) dropping all missing values from each feature and fitting the model, and (2) imputing missing values with the mean for continuous and discrete features and the mode for categorical features, then fitting the model. The performance of these strategies will be evaluated later in this report.
Finally, divided the training dataset into training and validation sets using the ‘train_test_split’ function.

**Learning algorithm(s)**
Several models were evaluated using the data. First, simple OLS regression was performed. This gave a near-zero MSE on the training set and an absurdly high MSE on the validation set, a display of pure overfitting. Then tried fitting a decision tree, which gave a MAE of 177 on the validation set. Also tried fitting a Random Forest Regressor for scikit-learn with several removed columns (“UserID”, “CurrentTask”, and “LastTaskCompleted”). This gave the MAE of 165 on the validation set; then added the columns “CurrentTask”, and “LastTaskCompleted” back and received the MAE of 161 on the validation set. A model based on Support Vector Machines was attempted to run, but even with the most computationally friendly hyperparameters, this model took too long to run, most likely owing to the data’s massive dimensionality.

**Ridge regression**
The best performing learning algorithm on the dataset was Ridge Regression, with an MSE of 96. This was later improved to 95 with post-processing the predictions in the following way: Since the dataset exhibits a large cluster of values at 1000, the predictions were altered such that any prediction above 920 was instead changed to 1000. This gave an MSE of 95.
Hyperparameter tuning consisted of changing the levels of this cutoff value, and changing the level of alpha, a hyperparameter of Ridge Regression. These efforts resulted in minor gains in MSE.

**Note**
The target values are not available in the test dataset because it was a competition, and the exact MSE for the test data of every model was computed by uploading the code to the competition server.
