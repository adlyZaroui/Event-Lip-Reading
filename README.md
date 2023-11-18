# Event Data Classification

This GitHub repository showcases an academic project that focuses on classifying event data obtained from an event-based sensor, also known as a neuromorphic sensor. The event data, provided by Prophesee, holds valuable insights that can be utilized across various domains.

Please note that the data used for this project will not be published in this repository due to ownership rights. However, the same dataset can be accessed and downloaded by anyone interested from the following Kaggle link: [Prophesee Dataset](https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022/data).

## Introduction

An event is a 4-tupe $(x,y,p,t)$ where

- $(x,y)$ represents the position of the pixel concerned by the event.
- $p$ is a boolen wether the change in luminosity is increasing or decreasing.
- $t$ represents the timestamp (in $\mu$s) since begining of the record.

Event Data are DataFrames where each row is an event, increasingly, *w.r.t.* timestamp.

## Usage

To use this project, follow these steps:

1. **Clone the repository**: First, clone this repository to your local machine using

    ```bash
    git clone https://github.com/adlyZaroui/Event-camera-classification.git
    ```

2. **Download the dataset**: The dataset can be downloaded from [this Kaggle link](https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022/data). Look for the folder named `train10` and download it. After downloading, place it in the `root` of your local repo.

This folder contains a folder also named `train10`, which contains 10 folders, at the end, it would looks like 

```bash
local_repo/
├──── train10/
│       ├── train10/
│         ├── Addition/
│         ├── Carnaval/
│         ├── Decider/
│         ├── Ecole/
│         ├── Fillette/
│         ├── Huitre/
│         ├── Joyeux/
│         ├── Musique/
│         ├── Pyjama/
│         └── Ruisseau/
├──── .venv/
├──── .gitignore
├──── .LICENSE
└──── *.ipynb
```

Each one of the folder in `train10/train10/` contains 32 csv files named from `0.csv` to `31.csv`, representing event data centered around the face of someone speaking, saying the specific word that is the parent folder name. (ex: in `train10/train10/Musique`, there is 32 csv files that are event data of someone saying the french world *Musique*).

More details can be find regarding the methodology used to record this dataset in the Kaggle Competiton link.

3. **Install dependencies**: This project requires certain Python libraries. You can install them using pip:

    ```bash
    pip install -r requirements.txt
    ```


The project starts with preprocessing the raw event data by converting it into pixel matrices, leveraging the inherent structure and characteristics of the data. This conversion enables the utilization of conventional image processing methods.

To handle high-dimensional data effectively, dimensionality reduction is employed using Principal Component Analysis (PCA). By reducing the dimensionality of the data while preserving relevant information, PCA allows for efficient computation and improved model performance.

The next step involves comparing the performance of a Random Forest classifier on the raw data versus the reduced data. Random Forest is a powerful ensemble learning algorithm known for its ability to handle high-dimensional data and deliver robust results. By evaluating the model performance on both versions of the data, valuable insights are gained regarding the impact of dimensionality reduction on classification accuracy.

Building upon these findings, a Bagging Random Forest classifier is constructed using the reduced data. Bagging enhances the predictive power of Random Forest by aggregating multiple decision trees trained on bootstrap samples of the dataset. This ensemble approach increases model stability, generalization, and overall classification accuracy.

Throughout this project, code samples, data preprocessing techniques, dimensionality reduction methods, model training, and evaluation procedures are provided, allowing researchers, data scientists, and machine learning enthusiasts to replicate the experiments and gain deeper insights into event data classification.

The repository also includes a detailed documentation, explaining the project's objectives, methodology, and results, along with relevant visualizations and performance metrics. By leveraging the power of Random Forest and dimensionality reduction techniques, this project offers valuable knowledge and resources for tackling event data classification challenges.

The Kaggle Challenge is available in the following link https://www.kaggle.com/competitions/smemi309-final-evaluation-challenge-2022

## Contributing:
Contributions to this project are highly encouraged and welcome. If you are interested in further enhancing the capabilities of event data classification, there are several areas where you can make valuable contributions. One possible area of contribution is to consider the raw data as time series and handle the problem as a multivariate time series classification.