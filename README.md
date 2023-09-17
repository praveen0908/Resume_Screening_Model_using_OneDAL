# RESUME SCREENING AND CLASSIFICATION MODEL

**GOAL**

Develop a model to classify resumes into predefined categories.
Note: Text classification is an example of supervised machine learning since we train the model with labeled data.

**DATASET**

The dataset used for this project is available in CSV format with 963 rows and 2 columns. You can access the dataset [here](https://www.kaggle.com/datasets/dhainjeamita/updatedresumedataset?resource=download).

**INTEL oneDAL LIBRARY**

To enhance the performance and efficiency of our model, we utilized the Intel oneDAL (one Data Analytics Library). oneDAL is a powerful library developed by Intel for high-performance data analytics and machine learning tasks. It provides a range of optimized algorithms and tools that significantly accelerate the processing of data.

![Intei oneAPI](https://github.com/praveen0908/Resume_Screening_Model_using_OneDAL/blob/main/INTEL%20oneDAAL.png)

**STEPS TAKEN**

All the required libraries and packages, including Intel oneDAL, were imported, and then the required dataset for the project was loaded.

*EDA* was carried out to visualize various parameters and the most correlated unigrams and bigrams.

Data was cleaned, also known as *Text Preprocessing*. Text Preprocessing was done using the re function of Python and the NLTK library, which is used for NLP-based models.

Model building was then implemented using different algorithms. We employed nine different models to train and evaluate the results, leveraging the power of the Intel oneDAL library for efficient computation.

![category_distribution](https://user-images.githubusercontent.com/86421205/184989201-89102de2-33d1-4472-85d1-8245280952ef.png)

**TEXT PREPROCESSING**

The text needed to be transformed into vectors so that the algorithms would be able to make predictions. In this case, the **Term Frequency – Inverse Document Frequency (TFIDF)** weight was used to evaluate how important a word is to a document in a collection of documents.

After removing punctuation and lowercasing the words, the importance of a word was determined in terms of its frequency, with the assistance of Intel oneDAL.

TF-IDF is a measure of the originality of a word.

**TF** is the number of times a term appears in a particular document.

**IDF** is a measure of how common or rare a term is across the entire corpus of documents.

![res](https://user-images.githubusercontent.com/86421205/184990238-7664e734-0e60-46a7-a778-f3fd79ebc2d5.png)

**MODELS USED**

The classification models used are:

1. *K Nearest Neighbor*
2. *Dummy Classifier*
3. *Linear Support Vector Classifier*
4. *Stochastic Gradient Descent*
5. *Random Forest*
6. *Decision Tree*
7. *Multinomial Naive Bayes Classifier*
8. *Gradient Boost*
9. *AdaBoost*

**LIBRARIES REQUIRED**

- Pandas: for data analysis
- Numpy: for data analysis
- Matplotlib: for data visualization
- Seaborn: for data visualization
- Scikit-learn: for data analysis
- Intel oneDAL: for enhanced performance and efficiency

**VISUALIZATION**

### Dataset Head snapshot
![data sample](https://user-images.githubusercontent.com/86421205/184983563-e11e69ab-266b-45ca-949c-68992b0a8dd5.png)

### Accuracy Comparison of Different models
![acuracy_comp(two)](https://user-images.githubusercontent.com/86421205/184983218-d01dba0d-98c0-4679-b08f-f2d65759df63.png)

### Evaluating SGD on different classes
![sgd](https://user-images.githubusercontent.com/86421205/184990143-e525fd7f-530c-4629-9f49-e5cb70668e17.png)

### Confusion matrix for Stochastic Gradient Descent Algorithm
![confusion_matrix_SGD](https://user-images.githubusercontent.com/86421205/184983825-5244289e-1583-4ac6-908d-fe0eb37bd7c9.png)

By viewing Confusion Matrix it is easily deduced that SGD model is the best model for this project.

**ACCURACIES**

| Model         | Architecture                      | Accuracy in % (on testing data) |
| ------------- |:---------------------------------:|:-------------:|
| Model 1       | K Nearest Neighbor Model          |97.92          |
| Model 2       | Dummy Classifier Model            |9.84           |
| Model 3       | Linear Support Vector Model       |100.00         |
| Model 4       | Stochastic Gradient Descent Model |100.00         |
| Model 5       | Random Forest Classifier Model    |100.00         |
| Model 6       | Decision Tree Classifier Model    |100.00         |
| Model 7       | Multinomial Naive Bayes Model     |96.37          |
| Model 8       | Gradient Boost Classifier Model   |100.00         |
| Model 9       | AdaBoost Model                    |30.05          |

**CONCLUSION**

The most successful model was found to be the Stochastic Gradient Descent Classifier for role classification based on resumes, with the significant support of the Intel oneDAL library to achieve efficient computation and optimization.


![Intel oneAPI](https://github.com/praveen0908/Resume_Screening_Model_using_OneDAL/blob/main/ONEAPI.png)
