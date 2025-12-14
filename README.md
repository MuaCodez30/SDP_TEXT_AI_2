# Fake News Detection using Machine Learning

A machine learning project that classifies news articles as either fake or true using Support Vector Machine (SVM) and Multi-Layer Perceptron (MLP) neural network models.

## 📋 Project Overview

This project implements a binary classification system to detect fake news articles. The system uses TF-IDF vectorization for feature extraction and compares the performance of two machine learning models: SVM and MLP.

## 🎯 Features

- **Text Preprocessing**: Cleans and normalizes text data by removing URLs, punctuation, digits, and extra whitespace
- **TF-IDF Vectorization**: Converts text into numerical features using Term Frequency-Inverse Document Frequency
- **Dual Model Approach**: Implements both SVM and MLP classifiers for comparison
- **Model Persistence**: Saves trained models and vectorizers for future use
- **Performance Metrics**: Generates detailed classification reports and confusion matrices

## 📊 Dataset

The dataset consists of:
- **Fake News**: 23,490 articles (labeled as 0)
- **True News**: 21,418 articles (labeled as 1)
- **Total**: 44,908 articles

Each article contains:
- `title`: Article headline
- `text`: Article content
- `subject`: Article category
- `date`: Publication date

## 📁 Project Structure

```
SDP_TEXT_2/
├── data/
│   ├── raw/
│   │   ├── Fake.csv          # Raw fake news dataset
│   │   └── True.csv          # Raw true news dataset
│   └── processed/
│       └── clean_fake_news_dataset.csv  # Preprocessed dataset
├── models/
│   ├── tfidf_vectorizer.pkl  # Saved TF-IDF vectorizer
│   ├── svm_model.pkl         # Trained SVM model
│   └── mlp_model.pkl         # Trained MLP model
├── results/
│   ├── metrics_SVC.csv       # SVM performance metrics
│   └── metrics_MLP.csv       # MLP performance metrics
└── scripts/
    ├── preprocess.ipynb      # Data preprocessing notebook
    └── train_model.ipynb     # Model training notebook
```

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (see `requirements.txt`)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SDP_TEXT_2
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

### Usage

#### 1. Data Preprocessing

Run the preprocessing notebook to clean and prepare the data:

```bash
jupyter notebook scripts/preprocess.ipynb
```

This notebook:
- Loads raw fake and true news datasets
- Combines and shuffles the data
- Cleans text by:
  - Converting to lowercase
  - Removing URLs and web links
  - Removing punctuation and special characters
  - Removing digits
  - Normalizing whitespace
- Saves the processed dataset to `data/processed/clean_fake_news_dataset.csv`

#### 2. Model Training

Run the training notebook to train and evaluate models:

```bash
jupyter notebook scripts/train_model.ipynb
```

This notebook:
- Loads the preprocessed dataset
- Applies TF-IDF vectorization (max_features=5000, ngram_range=(1,2))
- Splits data into train/test sets (80/20 split)
- Trains SVM and MLP models
- Evaluates model performance
- Saves models and metrics

## 🤖 Models

### Support Vector Machine (SVM)
- **Kernel**: Linear
- **Accuracy**: 99.57%
- **F1 Score**: 99.56%

### Multi-Layer Perceptron (MLP)
- **Architecture**: Hidden layers (128, 64)
- **Max Iterations**: 300
- **Accuracy**: 99.07%
- **F1 Score**: 99.04%

## 📈 Results

Both models achieve excellent performance on the test set:

| Model | Accuracy | F1 Score |
|-------|----------|----------|
| SVM   | 99.57%   | 99.56%   |
| MLP   | 99.07%   | 99.04%   |

The SVM model slightly outperforms the MLP model, with both showing high precision and recall for both classes.

## 🔧 Technical Details

### Text Preprocessing
- Lowercase conversion
- URL removal
- Punctuation removal
- Digit removal
- Whitespace normalization

### Feature Extraction
- **TF-IDF Vectorization**:
  - Maximum features: 5,000
  - N-gram range: (1, 2) - unigrams and bigrams
  - English stop words removal

### Model Configuration
- **Train/Test Split**: 80/20 with stratification
- **Random State**: 42 (for reproducibility)
- **SVM**: Linear kernel with probability estimates
- **MLP**: Two hidden layers (128, 64 neurons) with 300 max iterations

## 📦 Dependencies

See `requirements.txt` for the complete list of dependencies. Main packages include:
- pandas
- scikit-learn
- joblib
- jupyter

## 📝 Notes

- The dataset is balanced with a slight bias towards fake news (52.3% fake, 47.7% true)
- Models are saved using joblib for easy loading and inference
- All random operations use `random_state=42` for reproducibility

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for educational purposes.

## 👤 Author

Created as part of a Software Development Project (SDP).

---

**Note**: This project is for educational and research purposes. The models are trained on a specific dataset and may not generalize to all types of news articles.

