# Amazon Review Sentiment Analysis

## Project Overview

This project analyzes Amazon product reviews and classifies each review
into one of three sentiment categories: **Positive**, **Neutral**, or
**Negative**.

The workflow includes exploratory data analysis, sentiment-label
creation from star ratings, text preprocessing, TF-IDF feature
extraction, model training and evaluation, and an interactive Streamlit
app.

## Objectives

-   Explore and clean the review dataset.
-   Convert star ratings into sentiment labels.
-   Preprocess review text.
-   Convert text into numerical features using TF-IDF.
-   Train and compare machine-learning models.
-   Evaluate models with classification metrics.
-   Predict sentiment for a new review through a Streamlit interface.

## Dataset

The dataset contains these columns:

  Column     Description
  ---------- --------------
  `title`    Review title
  `rating`   Star rating
  `body`     Review text

Sentiment labels are derived from the rating:

  Rating   Sentiment
  -------- -----------
  1--2     Negative
  3        Neutral
  4--5     Positive

Place the dataset in the project directory before running the notebook.
If your file has a different name or different column names, update the
corresponding notebook cells.

## Workflow

1.  Load the dataset.
2.  Explore the data, missing values, duplicates, and rating/sentiment
    distributions.
3.  Create sentiment labels from ratings.
4.  Clean and preprocess review text.
5.  Convert text to TF-IDF features.
6.  Split data into stratified training and test sets.
7.  Train and compare classification models.
8.  Evaluate performance using accuracy, precision, recall, F1-score,
    and classification reports.
9.  Predict sentiment for new reviews.
10. Run the Streamlit application.

## Models Compared

-   Logistic Regression
-   Multinomial Naive Bayes
-   Linear Support Vector Machine (LinearSVC)
-   Random Forest
-   XGBoost

In the current notebook evaluation, **XGBoost achieved the highest
weighted F1-score (approximately 0.780)**. It tied Logistic Regression
on accuracy at approximately **80.2%** on the test split. Results may
change with different data, preprocessing, splits, or model settings.

The Neutral class is more difficult for the current model to identify
than the Positive and Negative classes. Improving Neutral-class recall
is an important next step.

## Technology Stack

-   Python and Jupyter Notebook
-   Pandas and NumPy
-   Matplotlib / Seaborn (where used for visualization)
-   NLTK
-   scikit-learn
-   XGBoost
-   Joblib
-   Streamlit

## Suggested Project Structure

Use the exact filenames present in your local project:

``` text
Sentiment_Analysis/
├── Sentiment_Analysis_Updated.ipynb
├── app.py
├── dataset.xlsx
├── xgboost_model(1).pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md
└── .gitignore
```

The notebook and Streamlit app should use compatible model, vectorizer,
and label-encoder files.

## Setup and Installation

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username:

``` bash
git clone https://github.com/YOUR_USERNAME/Sentiment_Analysis.git
cd Sentiment_Analysis
```

### 2. Create a virtual environment (recommended)

**Windows:**

``` bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**

``` bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

If you have a `requirements.txt` file:

``` bash
pip install -r requirements.txt
```

Otherwise, install the main packages:

``` bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn xgboost joblib streamlit openpyxl
```

If the saved scikit-learn artifacts were created with version **1.7.2**,
use the same version for compatibility:

``` bash
pip install scikit-learn==1.7.2
```

Saved model artifacts can be version-sensitive. If loading fails, check
the training and runtime library versions.

## Run the Notebook

Start Jupyter:

``` bash
jupyter notebook
```

Open `Sentiment_Analysis_Updated.ipynb`, verify the dataset path, and
run the cells from top to bottom.

## Run the Streamlit App

From the project directory, run:

``` bash
streamlit run app.py
```

Open the local URL shown in the terminal. Enter an Amazon review in the
app to see the predicted sentiment.

The app requires the model artifact, TF-IDF vectorizer, and label
encoder files configured in `app.py`.

## Evaluation Metrics

-   **Accuracy:** fraction of all predictions that are correct.
-   **Precision:** fraction of predicted examples for a class that are
    correct.
-   **Recall:** fraction of actual examples of a class that are
    correctly identified.
-   **F1-score:** harmonic mean of precision and recall.
-   **Weighted F1-score:** F1-score averaged across classes while
    accounting for class support.

For multi-class sentiment classification, inspect per-class metrics as
well as overall accuracy. A strong overall score can hide weak
performance on a particular class.

## Limitations and Future Improvements

-   Improve Neutral-class recall.
-   Test on additional unseen reviews.
-   Inspect misclassified examples and refine preprocessing.
-   Tune hyperparameters with cross-validation.
-   Explore class-weighting or resampling if appropriate.
-   Evaluate probability calibration before interpreting model scores as
    confidence.
-   Record dependency versions to improve reproducibility.

## Reproducibility and GitHub Notes

-   Use identical text preprocessing during training and prediction.
-   Keep the model, TF-IDF vectorizer, and label encoder consistent.
-   Record dependency versions in `requirements.txt`.
-   Do not commit credentials, private information, or virtual
    environments.
-   Check the dataset's license and sharing conditions before publishing
    it.

## Author

**Mahesh**\
Data Science \| Machine Learning \| Natural Language Processing

------------------------------------------------------------------------

If you find this project useful, feel free to star the repository or
share suggestions for improvement.
