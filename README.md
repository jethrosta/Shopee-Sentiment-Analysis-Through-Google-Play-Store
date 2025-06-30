# Shopee-Sentiment-Analysis-Through-Google-Play-Store

## 🧠 Project Overview
This project analyzes user reviews of the Shopee mobile app from the Google Play Store to classify sentiments as positive or negative. Implemented in Google Colab using Python, it includes data scraping, cleaning, preprocessing, TF‑IDF vectorization, and sentiment modeling via Machine Learning (Naive Bayes, Logistic Regression, etc.).

## 📋 Data Collection
Source: Scraped reviews from Google Play using Python (e.g., google-play-scraper).

Volume: Collected X,XXX reviews (after preprocessing: NNNN samples).

Labels: Reviews mapped to sentiment based on user ratings (>=4 stars → positive; <=2 stars → negative; 3-star reviews dropped or treated separately).

## 🔧 Data Preprocessing
1. Text cleaning: case folding, punctuation removal, URL and HTML filtering
2. Tokenization & normalization: hyphen removal, slang conversion, emoticon handling
3. Stopword removal (custom Indonesian-English stopwords)
4. Stemming using [a stemmer library like Sastrawi]
5. Final clean tokens ready for feature extraction

## 📈 Feature Engineering
Employed TF‑IDF vectorization

Tuned parameters such as ngram_range=(1,2) and max_df, min_df thresholds

Resulted in a sparse high-dimensional document-term matrix

## 🧩 Model Training & Evaluation
<div align="center">
  
| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | -------- | --------- | ------ | -------- |
| Naive Bayes         | 0.86     | 0.81      | 0.78   | 0.79     |
| Logistic Regression | …        | …         | …      | …        |
| (Others, if tested) | …        | …         | …      | …        |

</div>

## 🔬 Conclusion
That with the Shopee Review dataset from Google Playstore, it can experience overfitting because with 3 sentiment classes (positive, negative, and neutral) experiencing *data imbalance* so a method is needed so that the model is not dominant in the majority data. So I use **SMOTE** to do *oversampling*. This is reflected when all combinations are run without oversampling and all experience *overfitting* which is reflected in the accuracy of the testing data being more stagnant than the accuracy of the testing data. Then I determined the 3 best combinations from 12 combinations, namely TF-IDF + LSTM, TF-IDF + GRU, and N-gram + LSTM. Of these three combinations, when SMOTE was performed and optimization by finding the right hyperparameters was found to be the best in the **TF-IDF + GRU** combination where the results had **accuracy above 92%** aka **92.1%**. Then I found insight that actually my dataset is very simple so that using Multi Layer Perceptron + Optuna (to find the best parameters) has obtained accuracy above 92%. Therefore, for sentiment analysis projects, it seems unnecessary to use a deep learning algorithm that is too complex if the data is simple. Simply using a multilayer perceptron with a little optimization can already find high accuracy.

## 📊 Error Analysis
Identified common misclassifications (sarcasm, mixed sentiments, domain-specific slang). Learned preprocessing and model tuning could better handle these nuances

# 📝 Future Work
- Try advanced NLP techniques (e.g., fine-tuned IndoBERT)
- Increase dataset size with Google Play, social media, etc.
- Implement multilingual handling and emoji-aware tokenization
- Explore ensemble methods like Random Forest or deep learning (RNNs, BERT)
