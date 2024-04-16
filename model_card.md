# Breast Cancer Classification Model Card

## Model Details

- **Model Name**: Breast Cancer Classifier
- **Model Version**: 1.0
- **Model Type**: Supervised Machine Learning (Classification)
- **Model Purpose**: To predict whether a breast mass is malignant or benign based on features extracted from fine needle aspirate (FNA) images.
- **Model Author**: Indiwar Jha
- **Date Created**: 15/04/2024
- **Last Updated**: 15/04/2024

## Dataset Information

- **Dataset Name**: Breast Cancer Wisconsin (Diagnostic) Data Set
- **Dataset Source**: UCI Machine Learning Repository
- **Data Collection Method**: Features were computed from digitized images of FNA of breast masses.
- **Data Size**: 569 instances, 30 features
- **Target Variable**: Diagnosis (malignant or benign)

## Model Training

- **Preprocessing**: Standard scaling of features and removal of highly correlated columns.
- **Algorithms Used**: Logistic Regression, Support Vector Machine, Random Forest.
- **Hyperparameter Tuning**: Grid search with cross-validation was used to find the best hyperparameters for each algorithm.
- **Validation Set**: Used 20% of the training data for validation during hyperparameter tuning.

## Model Evaluation

- **Performance Metrics**:
  - Accuracy: It measures the proportion of correctly classified instances out of all instances in the dataset.
  - Precision: Precision measures the proportion of true positive predictions among all positive predictions made by the model. It is calculated as the ratio of true positives to the sum of true positives and false positives. Precision quantifies the model's ability to avoid false positives.
  - Recall: Recall, also known as sensitivity or true positive rate, measures the proportion of true positive predictions among all actual positive instances in the dataset. It is calculated as the ratio of true positives to the sum of true positives and false negatives. Recall quantifies the model's ability to capture all positive instances.
  - F1-score: F1-score is the harmonic mean of precision and recall. It provides a single metric that balances both precision and recall. F1-score is calculated as 2 * (precision * recall) / (precision + recall).
  - Confusion Matrix
- **Test Set Performance**:
  - Logistic Regression: Accuracy: 0.94

    |               | precision | recall | f1-score | support |
    |  malignant    |   0.97    |  0.86  |    0.91  |      43 |
    |  benign       |   0.92    |  0.99  |    0.95  |      71 |
    | ------------- | --------- | ------ | -------- | ------- |
    |  accuracy     |           |        |    0.94  |     114 |
    |  macro avg    |   0.95    |  0.92  |    0.93  |     114 |
    |  weighted avg |   0.94    |  0.94  |    0.94  |     114 |


  - Support Vector Machine: Accuracy = 0.96
    
    |               | precision | recall | f1-score | support |
    |  malignant    |   0.97    |  0.91  |    0.94  |      43 |
    |  benign       |   0.95    |  0.99  |    0.97  |      71 |
    | ------------- | --------- | ------ | -------- | ------- |
    |  accuracy     |           |        |    0.96  |     114 |
    |  macro avg    |   0.96    |  0.95  |    0.95  |     114 |
    |  weighted avg |   0.96    |  0.96  |    0.96  |     114 |

  - Random Forest: Accuracy = 0.92
    
    |               | precision | recall | f1-score | support |
    |  malignant    |   0.97    |  0.81  |    0.89  |      43 |
    |  benign       |   0.90    |  0.99  |    0.94  |      71 |
    | ------------- | --------- | ------ | -------- | ------- |
    |  accuracy     |           |        |    0.92  |     114 |
    |  macro avg    |   0.93    |  0.90  |    0.91  |     114 |
    |  weighted avg |   0.93    |  0.92  |    0.92  |     114 |

## Limitations

- Limited to the features provided in the dataset, which may not capture all aspects of breast mass characteristics.
- Performance may vary on unseen data or in different populations.

## Ethical Considerations

- **Bias**: The model's predictions may be influenced by biases present in the data.
- **Privacy**: No personally identifiable information was used in training or inference.
- **Fairness**: The model's predictions should be evaluated for fairness across different demographic groups.
- **Transparency**: Model internals and decision-making process should be made transparent to users.

## Potential Use Cases

- Early detection of breast cancer based on FNA images.
- Decision support for healthcare professionals in diagnosis and treatment planning.

## References

- UCI Machine Learning Repository: [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))