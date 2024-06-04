# IMDB Sentiment Analysis using various Bag-of-Words and Sequence to Sequence Models

## Project Overview
This project contains a comprehensive sentiment analysis on the IMDB movie reviews dataset. It employs various machine learning techniques, including both Bag of Words models and Sequence to Sequence models, to classify movie reviews as positive or negative. The Bag of Words models implemented include a Unigram model, which considers individual words as features; a Bigram model, which considers pairs of consecutive words; a Trigram model, which considers triplets of consecutive words; and a Bigram model enhanced with TF-IDF (term frequency-inverse document frequency). The Sequence to Sequence models encompass several approaches: one-hot encoded vectors, which provide a simple representation of words; word embeddings and pretrained embeddings using GloVe (Global Vectors for Word Representation); and transformers with positional embeddings, which represent state-of-the-art techniques in natural language processing. Additionally, a comparison of the accuracy of these models on test data is conducted and assessed, providing insights into their performance and effectiveness in sentiment classification. This diverse set of models allows for a thorough exploration of sentiment analysis on the IMDB dataset.

## Top 100 words by Frequency

<div>
  <img src='docs/top100words.png'>
</div>

## Bag-of-Words Models
Four distinct Bag-of-Words models are implemented, each utilizing different n-gram representations and techniques for text vectorization.

- The UniGram model serves as a baseline, employing individual words for representation and achieving an accuracy of 86.7% on the test data.
- The BiGram model considers pairs of consecutive words, enhancing performance to 89.5%.
- Further improvement is seen with the TriGram model, which captures sequences of three consecutive words, achieving 89.7% accuracy.
- A BiGram model incorporating TF-IDF representation is also introduced. Although it performs slightly worse than the other models, with an accuracy of 88.7%, it showcases the potential of combining n-gram representations with weighted term frequency information.

## Sequence to Sequence Models
Four distinct Sequence-to-Sequence models are implemented, each showcasing different architectures and techniques for text representation and processing:

- The One-Hot Encoded Vectors Model incorporates bidirectional LSTM layers for sequence processing. Despite its computational intensity, it achieves promising results at 85.7% accuracy, demonstrating the effectiveness of bidirectional processing.
- Models with Dense Embeddings utilize dense word embeddings to map integer tokens to dense vector representations. These models exhibit robust performance at 87.1% and lower computational demands compared to one-hot encoding.
- Pretrained GloVe Embeddings Models leverage pretrained embeddings to enhance the representation of input tokens. They showcase improved performance of 87.8% and reduced training time compared to models with randomly initialized embeddings.
- The Transformer Model with Positional Embedding showcases the effectiveness of attention mechanisms in capturing long-range dependencies and contextual information. The performance is 87.1%.

## Model Selection
Overall, the TriGram model exhibited the highest accuracy, highlighting the importance of capturing word sequences in sentiment analysis. Based on these results, the TriGram model was selected as the best-performing model. It will be used to predict sentiment on two random movie reviews.

## Predictions on Unseen Reviews
Sentiment prediction is demonstrated on three example sentences using a trained model. The sentences are tokenized into integer sequences using the same tokenizer used during training. The best-trained model, utilizing a 3-gram approach, is loaded for prediction. Predictions are made on the tokenized sequences, resulting in probability scores for positive and negative sentiment classes. A threshold of 0.5 is applied to convert the probability scores into binary classes: values above the threshold are classified as positive sentiment (1), while values below or equal to the threshold are classified as negative sentiment (0).

## Data Source:
https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz

## License:
This project is licensed under the Raza Mehar License. See the LICENSE.md file for details.

## Contact:
For any questions or clarifications, please contact Raza Mehar at [raza.mehar@gmail.com].
