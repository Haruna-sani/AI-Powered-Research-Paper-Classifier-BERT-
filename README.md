# AI-Powered Research Paper Classifier (BERT)

## Overview
This project implements a **BERT-based research paper classifier** that categorizes academic papers based on their abstracts. The model leverages **TensorFlow** and the **Hugging Face Transformers library** to fine-tune a pre-trained BERT model on the [ArXiv Scientific Research Papers Dataset](https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset).

## Dataset
The dataset used in this project consists of research paper abstracts labeled with their respective categories. It was sourced from Kaggle and contains thousands of research papers across multiple disciplines.

## Model Implementation
- **Preprocessing**: The dataset was tokenized using BERT's tokenizer, converting text into input IDs and attention masks.
- **Training**: The model was fine-tuned on the dataset using the **Adam optimizer** with a learning rate of **2e-5**.
- **Evaluation**: The model's performance was assessed using a confusion matrix and a classification report.

## Results
After training for a few epochs, the model attained an accuracy of **73%** on the test set. The classification report is as follows:

```
accuracy                           Accuracy: 0.73     
macro avg       Precision: 0.15      Recall: 0.11      F1-score: 0.11     
weighted avg    Precision: 0.70      Recall: 0.73      F1-score: 0.71     
```

## Analysis and Recommendations
### 1. **Data Imbalance Issue**
- The **macro average F1-score** is low, indicating an imbalance in class representation.
- The model struggles to classify underrepresented categories effectively.

**Recommendation:** Use **SMOTE (Synthetic Minority Over-sampling Technique)** or **class weighting** to handle data imbalance.

### 2. **Limited Training Time**
- The model was trained for a limited number of epochs.
- Further fine-tuning could improve performance.

**Recommendation:** Train for **more epochs** with early stopping to prevent overfitting.

### 3. **Alternative Model Architectures**
- BERT may not be the optimal model for this classification task.

**Recommendation:** Experiment with **RoBERTa** or **Longformer** for better context understanding in long texts.

## Future Work
- **Hyperparameter tuning** to optimize learning rates and batch sizes.
- **Data augmentation** to improve training data diversity.
- **Multi-label classification** for research papers that belong to multiple categories.

## Conclusion
This project successfully demonstrates the application of BERT for research paper classification. Although the model achieves **73% accuracy**, improvements can be made through further fine-tuning, handling class imbalance, and experimenting with different architectures.
## Data source : https://www.kaggle.com/datasets/sumitm004/arxiv-scientific-research-papers-dataset

