# beauty-skincare-customer-segmentation
Comparative Analysis of Clustering Algorithms for Customer Segmentation in Beauty and Skincare Retail.
# Customer Segmentation in Beauty and Skincare Retail

## Project Overview

This project focuses on customer segmentation in the beauty and skincare retail industry using Machine Learning clustering algorithms.

The main objective is to identify different customer groups based on their purchasing behaviour and characteristics. This helps businesses understand their customers and develop better marketing strategies.

## Objectives

- Analyse customer data to identify meaningful patterns.
- Apply different clustering algorithms for customer segmentation.
- Compare the performance of clustering algorithms.
- Identify the most suitable clustering algorithm using evaluation metrics.

## Dataset

- **Dataset:** Beauty and Skincare Customer Dataset
- **Records:** 1,000
- **File:** `Beauty_Skincare_Customer_Dataset_Research_1000.xlsx`

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Data preprocessing and visualization techniques

## Data Preprocessing

The following preprocessing techniques were applied:

- Missing categorical values were filled using mode.
- Missing numerical values were filled using mean.
- Label Encoding was used for categorical variables.
- StandardScaler was used for feature scaling.

## Machine Learning Algorithms

Three clustering algorithms were implemented and compared:

1. **K-Means Clustering**
2. **Agglomerative Clustering**
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**

## Model Evaluation

The clustering algorithms were evaluated using:

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index

## Results

K-Means clustering achieved the best overall performance among the three algorithms.

| Evaluation Metric | K-Means Result |
|---|---:|
| Silhouette Score | 0.3548 |
| Davies-Bouldin Index | 1.1272 |
| Calinski-Harabasz Index | 503.23 |
| Number of Clusters | 3 |

## Customer Segments

K-Means identified three customer segments:

| Cluster | Customers | Percentage |
|---|---:|---:|
| Cluster 0 | 424 | 42.4% |
| Cluster 1 | 250 | 25% |
| Cluster 2 | 326 | 32.6% |

These clusters represent different customer characteristics and purchasing patterns.

## Project Structure

- `SRC/` – Python source code
- `DATASET/` – Customer dataset
- `REPORTS/` – Project reports
- `RESULT/` – Project results and supporting documents

## Conclusion

This project demonstrates how clustering algorithms can be used to segment customers in the beauty and skincare retail industry.

Among the three algorithms, K-Means provided the best overall clustering performance. The identified customer segments can help businesses understand customer behaviour and make data-driven marketing decisions.

## Author

**MCA Student | Aspiring Software Developer & Data Analyst**

