import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import scikitplot as skplt

if __name__ == "__main__":
    st.title("Detecting Duplicate Data Analysis")
    uploaded_file = st.file_uploader("Choose a csv file")
    if uploaded_file is not None:
        """
        ### Contents
        1. Initial data checks and visualizations
        2. Data Preprocessing
        3. Modeling
            1. Baseline Accuracy
            2. Logistic Regression. Indented item
            3. Support Vector Machine
            4. Random Forest
            5. Multi Layer Perceptron
        4. Evaluation
        5. Next Steps
        """
        data = pd.read_csv(uploaded_file)
        inputs = data.iloc[:, :-1]
        inputs = (inputs - inputs.mean()) / inputs.std()
        targets = data.iloc[:, -1]
        colors = "flare"
        palette = sns.color_palette(colors)

        st.markdown("***")
        st.subheader("Selected Data")
        with st.expander("View data"):
            st.write(data)

        with st.expander("View summary statistics"):
            st.write("Summary Stats")
            st.write(data.describe())

            st.write("Correlation Matrix")
            st.write(data.corr())

        st.markdown("***")
        st.header("Exploratory Analysis")
        # ROW 1
        st.subheader("Target Classes")
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            totals = data["Classification"].value_counts()
            col1.metric(label='Total duplicate values', value=int(totals[0]))

        with col2:
            totals = data["Classification"].value_counts()
            col2.metric(label='Total distinct values', value=int(totals[1]))

        with col3:
            fig, ax = plt.subplots()
            ax = sns.barplot(x=data["Classification"].value_counts().index,
                             y=data["Classification"].value_counts(),
                             palette=palette[2:4])
            ax.set_title('Distinct and duplicate data counts')
            ax.set_xlabel('Class')
            ax.set_ylabel('Counts')
            st.pyplot(fig)

        # ROW 2 DISTRIBUTIONS
        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            fig, axes = plt.subplots(nrows=3, ncols=2)

            # pre-normalization plots
            fig.suptitle("Independent Variable Histograms")
            sns.histplot(data=data, x="Name_Score", kde=True, ax=axes[0, 0], color=palette[1])
            axes[0, 0].set(xlabel='Name Score', ylabel="")
            sns.histplot(data=data, x="Address_Score", kde=True, ax=axes[0, 1], color=palette[1])
            axes[0, 1].set(xlabel='Address Score', ylabel="")
            sns.histplot(data=data, x="City_Score", kde=True, ax=axes[1, 0], color=palette[1])
            axes[1, 0].set(xlabel='City Score')
            sns.histplot(data=data, x="URL_Score", kde=True, ax=axes[1, 1], color=palette[1])
            axes[1, 1].set(xlabel='URL Score', ylabel="")
            sns.histplot(data=data, x="Phone_Score", kde=True, ax=axes[2, 0], color=palette[1])
            axes[2, 0].set(xlabel='Phone Score')
            fig.tight_layout()
            axes[2, 1].set_visible(False)
            st.pyplot(fig)
        with col2:
            st.subheader("Input Data Distribution Plots")
            """
            Histogram plots for the independent variables show that these data most closely follow uniform
            distributions. 
            """

        # ROW 3
        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            """
            ### Dimensionality Reduction with T-SNE
            The T-SNE embedding does not show strong clustering. Dimensionality reduction will not likely be a useful 
            feature extraction step for these data.   
            """

        with col2:
            tsne = TSNE(n_components=2, init='pca', random_state=0)
            # fit data with TSNE
            fit = tsne.fit_transform(data.iloc[:, :-1])
            # Plot fit results
            fig, ax = plt.subplots()
            ax = sns.scatterplot(x=fit[:, 0], y=fit[:, 1], hue=data["Classification"], palette=palette[2:4])
            st.pyplot(fig)

        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            fig, axes = plt.subplots(nrows=3, ncols=2)
            # post-normalization plots
            fig.suptitle("Independent Variable Distributions (normalized)")
            sns.histplot(data=inputs, x="Name_Score", kde=True, ax=axes[0, 0], color=palette[4])
            sns.histplot(data=inputs, x="Address_Score", kde=True, ax=axes[0, 1], color=palette[4])
            axes[0, 1].set(xlabel='Address Score', ylabel="")
            sns.histplot(data=inputs, x="City_Score", kde=True, ax=axes[1, 0], color=palette[4])
            axes[1, 0].set(xlabel='City Score')
            sns.histplot(data=inputs, x="URL_Score", kde=True, ax=axes[1, 1], color=palette[4])
            axes[1, 1].set(xlabel='URL Score', ylabel="")
            sns.histplot(data=inputs, x="Phone_Score", kde=True, ax=axes[2, 0], color=palette[4])
            axes[2, 0].set(xlabel='Phone Score')
            fig.tight_layout()
            axes[2, 1].set_visible(False)
            st.pyplot(fig)

        with col2:
            """
            ### Data Pre Processing
            There aren't many preprocessing steps needed for these data. The means and standard deviations 
            for each feature are close and there are no outliers or null values that need to be removed. All of 
            the features have meaningful names and the data types are consistent in the dataframe. The data will still 
            be normalized to mean=0 and sd=1 to improve numerical stability. 
            """

        # ROW 5 MODELS
        ## Example
        st.markdown("***")
        st.header("Modeling")
        col1, col2 = st.columns(2)
        with col1:
            total_observations = int(len(inputs))
            total_correct = int(targets.value_counts()[0])
            col1.metric(label='Baseline Accuracy', value=round(total_correct / total_observations, 3))

        with col2:
            """
            ### Baseline Accuracy
            A baseline for comparing model accuracy can be established by looking at the
            results of a model which predicts the mode value for every input.
            """

        # ROW 6 LR
        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            """
            ### Logistic Regression
            Logistic regression is the best place to start with these data. The modeling assumptions
            are met and it is a simple model. since it is deterministic, it will be a useful benchmark for
            comparing other models. 
            """

        with col2:
            LR = LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr').fit(inputs, targets)
            col1.metric(label='Logistic Regression Mean Accuracy', value=round(LR.score(inputs, targets), 2))
            fig = plt.figure(figsize=(15, 6))
            Y_pred = LR.predict(inputs)
            col1.metric(label="F1 Score", value=round(f1_score(targets, Y_pred, pos_label="Distinct"), 3))

            fig, ax = plt.subplots()
            skplt.metrics.plot_confusion_matrix(targets,
                                                Y_pred,
                                                title="Confusion Matrix",
                                                cmap=sns.color_palette(colors, as_cmap=True),
                                                ax=ax)
            st.pyplot(fig)

        # ROW 6 SVM
        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            """
            ### Support Vector Machine
            Another simple model that can be used for binary classification is SVM. If the data is linearly
            separable, SVM will find a model with perfect prediction. Based on the T-SNE plot, this is unlikely
            but the SVM will still be a useful benchmark to test other models against since it is a deterministic
            model. 
            """

        with col2:
            SVM = svm.LinearSVC()
            SVM.fit(inputs, targets)
            col1.metric(label='SVM Mean Accuracy', value=round(SVM.score(inputs, targets), 3))
            y_pred = SVM.predict(inputs)
            col1.metric(label="F1 Score", value=round(f1_score(targets, Y_pred, pos_label="Distinct"), 3))

            fig, ax = plt.subplots()
            skplt.metrics.plot_confusion_matrix(targets,
                                                y_pred,
                                                title="Confusion Matrix",
                                                cmap=sns.color_palette(colors, as_cmap=True),
                                                ax=ax)
            st.pyplot(fig)


        # ROW 7 RF AND MLP
        st.markdown("***")
        col1, col2 = st.columns(2)

        with col1:
            """
            ### Random Forest
            The random forest model has high accuracy with these data. OOB score shows individual estimators do well
            on unseen data. Test score also indicates strong prediction performance on unseen data.

            * Train/Test split: 60/40
            """

        with col2:
            n_estimators = st.slider('Number of trees for random forest', 1, 8, 8)
            max_depth = st.slider('Max depth for trees', 2, 4, 3)
            RF = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=0, oob_score=True)
            X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.4)
            RF.fit(X_train, y_train)
            y_pred = RF.predict(X_test)

            col1.metric(label='Random Forest Test Accuracy', value=round(RF.score(X_test, y_test), 2))
            col1.metric(label='Random Forest OOB score', value=round(RF.oob_score_, 2))
            col1.metric(label="F1 Score", value=round(f1_score(y_test, y_pred, pos_label="Distinct"), 3))

            fig, ax = plt.subplots()
            skplt.metrics.plot_confusion_matrix(y_test,
                                                y_pred,
                                                title="Confusion Matrix",
                                                cmap=sns.color_palette(colors, as_cmap=True),
                                                ax=ax)
            st.pyplot(fig)


        st.markdown("***")
        col1, col2 = st.columns(2)
        with col1:
            """
            ### Multi Layer Perceptron
            The MLP has high accuracy on these data but variation in the results can be seen 
            due to the stochastic algorithm. 

            * Train/Test/Validate split: 50/40/10
            * Optimizer: Adam
            * Activation: ReLU

            """

        with col2:
            n_hidden_units = st.slider('Number of nodes per hidden layer', 2, 8, 5)
            n_layers = st.slider('Number of hidden layers', 2, 4, 2)
            NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(n_hidden_units, n_layers), random_state=1)
            X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.4)
            NN.fit(X_train, y_train)
            y_pred = NN.predict(X_test)
            col1.metric(label='NN Test Mean Accuracy', value=round(NN.score(X_test, y_test), 3))
            col1.metric(label='NN Train Mean Accuracy', value=round(NN.score(X_train, y_train), 3))
            col1.metric(label="F1 Score", value=round(f1_score(y_test, y_pred, pos_label="Distinct"), 3))

            fig, ax = plt.subplots()
            skplt.metrics.plot_confusion_matrix(y_test,
                                                y_pred,
                                                title="Confusion Matrix",
                                                cmap=sns.color_palette(colors, as_cmap=True),
                                                ax=ax)
            st.pyplot(fig)

    else:
        st.write("no data to analyze")
