# Event Data Classification using Random Forest and Dimensionality Reduction

This GitHub repository presents a comprehensive project that focuses on classifying event data obtained from event-based sensors. Event data, provided by Prophesee, holds valuable insights that can be utilized in a multitude of domains.

The project starts with preprocessing the raw event data by converting it into pixel matrices, leveraging the inherent structure and characteristics of the data. This conversion enables the utilization of conventional image processing methods.

To handle high-dimensional data effectively, dimensionality reduction is employed using Principal Component Analysis (PCA). By reducing the dimensionality of the data while preserving relevant information, PCA allows for efficient computation and improved model performance.

The next step involves comparing the performance of a Random Forest classifier on the raw data versus the reduced data. Random Forest is a powerful ensemble learning algorithm known for its ability to handle high-dimensional data and deliver robust results. By evaluating the model performance on both versions of the data, valuable insights are gained regarding the impact of dimensionality reduction on classification accuracy.

Building upon these findings, a Bagging Random Forest classifier is constructed using the reduced data. Bagging enhances the predictive power of Random Forest by aggregating multiple decision trees trained on bootstrap samples of the dataset. This ensemble approach increases model stability, generalization, and overall classification accuracy.

Throughout this project, code samples, data preprocessing techniques, dimensionality reduction methods, model training, and evaluation procedures are provided, allowing researchers, data scientists, and machine learning enthusiasts to replicate the experiments and gain deeper insights into event data classification.

The repository also includes a detailed documentation, explaining the project's objectives, methodology, and results, along with relevant visualizations and performance metrics. By leveraging the power of Random Forest and dimensionality reduction techniques, this project offers valuable knowledge and resources for tackling event data classification challenges.

The Kaggle Challenge is available in the following link https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022
