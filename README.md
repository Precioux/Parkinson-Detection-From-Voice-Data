
# Parkinson's Disease Detection Using Machine Learning

This project aims to develop a machine learning model to predict the presence of Parkinson's disease based on voice recordings. Parkinson's disease is a neurodegenerative disorder that affects motor skills, and certain voice patterns can be indicative of the condition. By analyzing features extracted from voice data, the model can help in early detection of the disease.

![Parkinson's Detection](https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMSkillsNetwork-GPXX01JCEN/images/parkinson_patient_voice.png)

## Project Overview

The goal of this project is to use supervised machine learning algorithms to detect Parkinson's disease from voice recordings. The data contains various features extracted from voice recordings, such as pitch, jitter, shimmer, and other vocal attributes. These features are used to train a classification model to predict whether the individual is affected by Parkinson's disease.

### Key Features:
- Machine learning for binary classification of Parkinson’s disease.
- Data pre-processing and feature engineering for voice data.
- Visualization of model performance using metrics like accuracy, precision, recall, and F1-score.

## Setup

### Prerequisites

Before running the project, ensure you have the following installed:
- Python 3.x
- Scikit-learn
- Seaborn
- dtreeviz (for decision tree visualizations)
- Other necessary dependencies listed in the `requirements.txt` file.

### Installing Required Libraries

You can install the required libraries by running the following command:

```bash
pip install -r requirements.txt
```

Or manually install the key libraries:

```bash
pip install scikit-learn==1.2.1 seaborn dtreeviz
```

## Dataset

The dataset used in this project consists of voice recordings from individuals, with various features extracted to serve as inputs to the machine learning models. These features include metrics such as:
- Jitter: Variability in frequency
- Shimmer: Variability in amplitude
- Harmonics-to-Noise Ratio (HNR)
- Fundamental frequency (Pitch)

The target variable is a binary label indicating whether the individual has Parkinson’s disease.

### Preprocessing

The dataset is preprocessed by normalizing the features and handling any missing values. Feature selection techniques may be applied to improve the model's performance.

## Model

Several machine learning models are applied to this dataset, including:
- Decision Trees
- Support Vector Machines (SVM)
- Random Forest Classifiers
- Other possible classifiers for comparison

The model is evaluated using performance metrics such as accuracy, precision, recall, and F1-score. Visualization of decision trees is done using the `dtreeviz` library.

## Training

To train the model, follow the steps outlined in the notebook:
1. Load the dataset.
2. Preprocess the data.
3. Train the machine learning model.
4. Evaluate model performance on the test set.

## Results

The notebook includes model evaluation metrics and visualizations to help interpret the model's predictions. Key results include:
- Confusion matrix
- ROC curve and AUC score
- Decision tree visualization

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/parkinson-detection.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook to preprocess the data, train the model, and evaluate its performance.

## Future Improvements

- Implement more advanced machine learning techniques like ensemble learning or deep learning.
- Explore additional features or data sources for improved predictions.
- Deploy the model as a web application or service for real-time predictions.

## Contributing

Feel free to open issues or pull requests if you'd like to contribute to the project.

## License

This project is licensed under the MIT License.
