# IRTM

NTU 110-1 Information Retrieval and Text Mining (#IM5030)

## Assignment overview
|  Assignment  |  Description  |
|  -    |       -       |
| pa1 | Basic pre-processing of a text, such as Tokenization, lowercase conversion, and Stemming. |
| pa2 | (1) Create a dictionary of 1095 news articles and calculate the normalized tf-idf value of each term.<br>(2) Write a function to calculate cosine similarity for two articles, which will be used in Assignment 4. |
| pa3 | Supervised classification of 1095 news items. <br> (1) 13 categories with 195 news items were used for training. <br> (2) Feature Selection is performed using Chi-square, and Multinomial Naive Bayes Classifier is used for classification. <br> (3) Finally, I uploaded the classification results to Kaggle, and got a score of =0.99333= for this job, ranking 4/75 in the class. |
| pa4 | Unsupervised clustering (k=8, 13, 20) was performed on 1095 news items. <br> (1) The pair-wise cosine similarity is calculated using normalized tf-idf values, and the bottom-up HAC is used to perform the binning with Max Heap acceleration. <br> (2) In each iteration of HAC, the similarity is measured using single-link. |

## Subfolder Architecture
Each assignment subfolder will probably have these things: documentation (task descriptions and submitted reports), code(.py), input data and output data

```bash
pa1
|- report.pdf
|- pa1.py
|- input data: e.g. 1095 news, training data list
|- output data: e.g. dictionary.txt
```