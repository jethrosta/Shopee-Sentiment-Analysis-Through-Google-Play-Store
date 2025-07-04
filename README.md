# Shopee Customer Sentiment Analysis from Google Play Store Reviews

<div>
  <img src = "https://github.com/jethrosta/Shopee-Sentiment-Analysis-Through-Google-Play-Store/blob/main/images/wordCloud.png">
</div>

## 🎯 Business Problem & Project Goal

This project aims to analyze user sentiment towards the Shopee application by examining reviews from the Google Play Store. The primary goal is to build a robust sentiment classification model that can automatically categorize reviews as **positive**, **negative**, or **neutral**.

The insights from this analysis can help Shopee's product, marketing, and customer service teams to:
- Identify key drivers of customer satisfaction and dissatisfaction.
- Prioritize areas for app improvement and feature development.
- Monitor brand health and public perception in near real-time.
- Make data-driven decisions to enhance the overall user experience.

## 📊 Key Insights & Actionable Recommendations

### 1. Sentiment Distribution & Data Imbalance
Initial analysis of 17,874 user reviews revealed a significant data imbalance. The dataset was heavily skewed towards positive and negative reviews, with very few neutral ones:
- Positive: 8,527 reviews
- Negative: 8,203 reviews
- Neutral: 1,144 reviews

<div align = "center">
<img src ="https://github.com/jethrosta/Shopee-Sentiment-Analysis-Through-Google-Play-Store/blob/main/images/distribusiSentimen.png">
</div>

**Finding**: A model trained on this imbalanced data would likely be biased and perform poorly, especially in identifying neutral feedback. To counteract this, the **SMOTE** (Synthetic Minority Over-sampling Technique) was applied, which successfully balanced the dataset by creating synthetic samples for the minority class. This was a critical step to prevent model overfitting and ensure reliable performance.

### 2. Model Complexity vs. Performance
While multiple complex deep learning architectures were tested, a key finding was that a simpler model could achieve top-tier results.

  **Finding**: After hyperparameter tuning with Optuna, a relatively simple Multilayer Perceptron (MLP) model achieved an accuracy of over 92%. This was comparable to the best-performing complex model, a GRU (Gated Recurrent Unit) with TF-IDF features (92.1% accuracy).

<div align="center">
  <img src="https://github.com/jethrosta/Shopee-Sentiment-Analysis-Through-Google-Play-Store/blob/main/images/maxresdefault.jpg">
  <img src="https://github.com/user-attachments/assets/f6c85501-cec8-451b-bb25-c38b9944005d">
</div>

## ⚙️ Analytical Workflow & Methodology
### 1. Data Collection
- Source: Google Play Store
- Tool: google-play-scraper Python library
- Volume: 20,000 raw reviews scraped. After cleaning and preprocessing, 17,874 unique and complete reviews were used for the analysis.

### 2. Data Preprocessing
A comprehensive text cleaning pipeline was implemented to prepare the Indonesian-language reviews for modeling. This involved:
- Text Cleaning: Removal of mentions, hashtags, URLs, numbers, and punctuation.
- Normalization: Conversion to lowercase and correction of common slang words (e.g., "gak" → "tidak").
- Tokenization: Splitting text into individual words.
- Stopword Removal: Filtering out common Indonesian and English stopwords (e.g., "di", "dan", "the") using a custom list.

### 3. Feature Engineering
- TF-IDF (Term Frequency-Inverse Document Frequency): This was the primary method used to create a numerical matrix. It captures the importance of a word not just by its frequency in a single review (Term Frequency) but also by how unique it is across all reviews (Inverse Document Frequency). This helps to prioritize words that are more specific and informative.
- Bag of Words(BoW) : This model represents text by counting the occurrence of each word, effectively creating a "bag" of words without considering grammar or word order. It's a simple yet effective way to represent text data based purely on word frequency.
- Word2Vec : This technique creates dense vector representations of words (word embeddings). Unlike TF-IDF or BoW, Word2Vec captures the context and semantic relationships between words. For example, it can learn that "diskon" (discount) and "promo" (promotion) are similar concepts.
- N-gram: In addition to single words, combinations of words (specifically bigrams, or pairs of two words) were tested. This provides the models with more contextual information (e.g., "gratis ongkir" - free shipping) that would be lost if only single words were considered.

## 📈 Model Training & Performance
Several machine learning and deep learning models were trained and evaluated. To address the class imbalance noted earlier, models were trained on data that had been resampled using SMOTE. The top-performing combination was a GRU model with TF-IDF features, which was further optimized using Optuna to find the best hyperparameters.

<div align="center">
  
| Model                    | Accuracy | Loss | Validation Accuracy | Validation Loss|
| -------------------      | -------- | -----| ------------------- | ---------------|
| TF-IDF + LSTM(Optimized) | 92%      | 26%  | 92%                 | 36%            | 
| **TF-IDF + GRU(Optimized)**  | **92%**      | **23%**  | **92%**                 | **32%**            |
| N-gram + LSTM(Optimized) | 85%      | 42%  | 85%                 | 50%            |

*Highest Accuracy:*
<div>
  <img src="https://github.com/jethrosta/Shopee-Sentiment-Analysis-Through-Google-Play-Store/blob/main/images/TF-IDF_GRU.png">
</div>

</div>

# 🔬 Conclusion
The Shopee review dataset from the Google Play Store was prone to overfitting. This was because the three sentiment classes (positive, negative, and neutral) suffered from a data imbalance, which required a method to prevent the model from being biased towards the majority class. Therefore, I used SMOTE to perform oversampling. This issue became clear when all model combinations were run without oversampling; they all experienced overfitting, which was reflected by the testing accuracy being more stagnant than the training accuracy.

Subsequently, I identified the 3 best combinations out of 12 that were tested: TF-IDF + LSTM, TF-IDF + GRU, and N-gram + LSTM. Among these three, after applying SMOTE and optimizing to find the right hyperparameters, the best performance was achieved with the **TF-IDF + GRU combination**, which resulted in an accuracy of over 92%, specifically 92.1%.

Later, I discovered an insight that my dataset is actually quite simple. Using a Multi-Layer Perceptron + Optuna (to find the best parameters) also achieved an accuracy of over 92%. Therefore, for sentiment analysis projects with simple data, it seems unnecessary to use an overly complex deep learning algorithm. A high level of accuracy can be achieved simply by using a multilayer perceptron with some optimization.

# 💡 Recommendations for Shopee
- For the Data Science Team: For this specific sentiment analysis task, a complex deep learning model is not necessarily required. An optimized MLP can provide high accuracy with significantly lower computational costs and faster training times, making it a more efficient solution for production environments.
- For the Product Team: A deeper dive into the content of the misclassified reviews (error analysis) could reveal nuanced feedback, such as sarcasm or domain-specific slang, that the current model struggles with. Addressing these areas in future preprocessing steps could further improve accuracy.
