import matplotlib.pyplot
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import streamlit as st
from pysondb import db


if __name__ == "__main__":
    data = pd.read_csv(Path('../data/duplicate_data_case_study.csv'))

    # check first 5 rows to make sure dataframe looks correct
    print(data.head())

    # check data summary stats and general info
    print(data.info())
    print(data.describe())

    # select input data columns
    inputs = data.iloc[:, :-1]
    print(inputs.head())

    # select targets column
    targets = data.iloc[:, -1]
    print(targets.head())

    # normalize input to data to mean=0, sd=1
    inputs = (inputs-inputs.mean())/inputs.std()
    print(inputs.info())
    print(inputs.describe())
    # for row in data.iterrows():

    # show barplot of class counts
    sns.barplot(x=data["Classification"].value_counts().index, y=data["Classification"].value_counts())
    plt.title('Distinct and duplicate data counts')
    plt.xlabel('Class')
    plt.ylabel('Counts')
    plt.show()
    #
    palette = "Set2"
    fig, axes = plt.subplots(nrows=3, ncols=2)

    # pre-normalization plots
    fig.suptitle("Independent Variables Histograms")
    sns.histplot(data=data, x="Name_Score", kde=True, ax=axes[0, 0], bins=50)
    axes[0, 0].set(xlabel='Name Score', ylabel="")
    sns.histplot(data=data, x="Address_Score", kde=True, ax=axes[0, 1])
    axes[0, 1].set(xlabel='Address Score', ylabel="")
    sns.histplot(data=data, x="City_Score", kde=True, ax=axes[1, 0])
    axes[1, 0].set(xlabel='City Score')
    sns.histplot(data=data, x="URL_Score", kde=True, ax=axes[1, 1])
    axes[1, 1].set(xlabel='URL Score', ylabel="")
    sns.histplot(data=data, x="Phone_Score", kde=True, ax=axes[2, 0])
    axes[2, 0].set(xlabel='Phone Score')
    fig.tight_layout()
    axes[2, 1].set_visible(False)
    plt.show()

    fig, axes = plt.subplots(nrows=3, ncols=2)
    # post-normalization plots
    fig.suptitle("Independent Variable Distributions (normalized)")
    sns.histplot(data=inputs, x="Name_Score", kde=True, ax=axes[0, 0])
    sns.histplot(data=inputs, x="Address_Score", kde=True, ax=axes[0, 1])
    axes[0, 1].set(xlabel='Address Score', ylabel="")
    sns.histplot(data=inputs, x="City_Score", kde=True, ax=axes[1, 0])
    axes[1, 0].set(xlabel='City Score')
    sns.histplot(data=inputs, x="URL_Score", kde=True, ax=axes[1, 1])
    axes[1, 1].set(xlabel='URL Score', ylabel="")
    sns.histplot(data=inputs, x="Phone_Score", kde=True, ax=axes[2, 0])
    axes[2, 0].set(xlabel='Phone Score')
    fig.tight_layout()
    axes[2, 1].set_visible(False)
    plt.show()

    # instantiate TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    # fit data with TSNE
    fit = tsne.fit_transform(data.iloc[:, :-1])
    # Plot fit results
    sns.scatterplot(x=fit[:, 0], y=fit[:, 1], hue=data["Classification"], palette="Set2")
    plt.show()