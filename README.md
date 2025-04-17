# Emotion Detector for Twitter Captions 
[Link to Colab Notebook]([https://example.com](https://colab.research.google.com/drive/1Q1OSkETsPRJlEtC-3sYsWlZM5k9_N6zq?usp=sharing))

This project develops an emotion classification system for Twitter captions using DistilBERT to identify six emotions: sadness, joy, love, anger, fear, and surprise. It utilizes the Hugging Face ecosystem (`Datasets`, `Tokenizers`, `Transformers`) to process the "emotion" dataset and implement two approaches:

- **Dataset Preparation**:

  - Loads the "emotion" dataset (16,000 training samples) with tweets labeled across six emotions.
  - Tokenizes text using DistilBERT’s tokenizer, applying padding and truncation for model compatibility.

- **Feature Extraction Approach**:

  - Extracts hidden states from DistilBERT as fixed features (768-dimensional vectors per tweet).
  - Trains a logistic regression classifier on these features, achieving 63.3% accuracy on the validation set.
  - Visualizes embeddings with UMAP, revealing overlapping emotions like sadness, anger, and fear.

- **Fine-Tuning Approach**:

  - Fine-tunes the entire DistilBERT model with a classification head using `Trainer` and `TrainingArguments`.
  - Achieves 92.25% accuracy and F1-score on the validation set after 2 epochs.
  - Confusion matrix shows improved performance, though love is often confused with joy, and surprise with fear/joy.
