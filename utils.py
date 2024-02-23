# Importing necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as clr

# Metrics, tools etc.
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_validate


class AnalyserUtil:
    """
    Analyse helper functions for notebook in exploratory data analysis (eda) step.
    """
    def check_df(self,dataframe:pd.DataFrame, head=5, tail=5):
        """
        Starts EDA using this function to see whole picture of the dataframe.
        Prints columns and their dtypes, general shape, head & tail, count of null & duplicated values
        and final with descriptive statistics.

        Parameters
        ----------
        dataframe: pd.DataFrame
            Dataframe to be observed in big picture.
        head: int, default=5
            To observe first n observations in the dataframe.
        tail: int, default=5
            To observe last n observations in the dataframe.
        """
        print(" COLUMNS ".center(60, "*"))
        print(dataframe.columns)
        print(" INFO ".center(60,"*"))
        print(dataframe.info(memory_usage="deep"))
        print(" SHAPE ".center(60, "*"))
        print(f"Shape of dataframe: {dataframe.shape}")
        print(f"# of observations: {dataframe.shape[0]}")
        print(f"# of variables: {dataframe.shape[1]}")
        print(" HEAD & TAIL ".center(60, "*"))
        print(f" First {head} observations ".center(40, "*"))
        print(dataframe.head(head))
        print(f" Last {tail} observations ".center(40, "*"))
        print(dataframe.tail(tail))
        print(" NULL VALUES ".center(60, "*"))
        print(dataframe.isna().sum())
        print(" DUPLICATED ROWS ".center(60, "*"))
        print(f"# of duplicated rows: {dataframe[dataframe.duplicated()].shape[0]}")
        print(" DESCRIBE ".center(60, "*"))
        print(dataframe.describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.95]).T)

    def cat_summary(self,dataframe: pd.DataFrame, catcol:str):
        """
        Summarizes categorical variables by plots consisting of barplot which shows counts and pie chart
        which shows ratios. After these plots, shows the situation against to our target variable 'exited'
        using groupby.

        Parameters
        ----------
        dataframe: pd.DataFrame
        catcol: str
            Categorical variable name from the dataframe to be observed.
        """
        plt.suptitle(f"{catcol.upper()}")
        plt.subplot(1, 2, 1)
        counts = dataframe[catcol].value_counts().reset_index()
        ax = sns.barplot(counts,
                         x="index",
                         y=catcol,
                         hue="index",
                         legend=False)
        for i in range(dataframe[catcol].nunique()):
            ax.bar_label(ax.containers[i])

        plt.subplot(1, 2, 2)
        plt.pie(counts[catcol],
                labels=counts["index"],
                autopct="%1.1f%%",
                colors=list(reversed(clr.TABLEAU_COLORS.values())))
        plt.show(block=True)

        if "exited" != catcol:
            print(dataframe.groupby(catcol)["exited"].mean(), end="\n\n")

    def num_summary(self,dataframe: pd.DataFrame, numcol:str):
        """
        Summarizes numerical variables by plots consisting of histplot which shows distributions and boxplot
        which is useful for descriptive statistics. After these plots, shows the situation of our target variable
        against to numerical variables.

        Parameters
        ----------
        dataframe: pd.DataFrame
        numcol: str
            Numerical variable name from the dataframe to be observed.
        """
        plt.suptitle(f"{numcol.upper()}")
        plt.subplot(1, 2, 1)
        sns.histplot(dataframe, x=numcol)
        plt.xlabel("Distribution")
        plt.ylabel("")

        plt.subplot(1, 2, 2)
        sns.boxplot(dataframe, y=numcol)
        plt.ylabel("")
        plt.xlabel("Boxplot")
        plt.show(block=True)

        print(dataframe.groupby("exited")[numcol].mean(), end="\n\n")

    def correlation_matrix(self,dataframe: pd.DataFrame,numcols:list):
        """
        Correlation analysis plotting correlation matrix of the numerical variables of the dataframe.

        Parameters
        ----------
        dataframe: pd.DataFrame
        numcols: list
            Numerical variables of the dataframe
        """
        corr = dataframe[numcols].corr()
        sns.heatmap(corr,
                    mask=np.triu(corr),
                    annot=True,
                    cmap="RdBu", linewidths=0.3)
        sns.set_style("dark")
        plt.show(block=True)


class FeatureEngineeringUtil:
    """
    Feature engineering helper functions.
    """
    def grabbing_vars(self,dataframe: pd.DataFrame, cat_obj_th=10, cat_num_th=5, summary=False):
        """
        Grabs variables from the dataframe easily according to thresholds. Especially useful for
        feature engineering.

        Parameters
        ----------
        dataframe: pd.DataFrame
        cat_obj_th: int, default=10
            Threshold of object categoric variables such as color, gender etc.
        cat_num_th: int, default=5
            Threshold of numeric categoric variables such as classes, degree of family etc.
        summary: bool, default=False
            If true, prints number of each type of variables and their variable lists. It is good for using
            in notebook.

        Returns
        -------
        obj_list: list
            List of object variables
        num_list: list
            List of numeric variables
        cat_list: list
            List of categoric variables
        """
        obj_vars = [col for col in dataframe.select_dtypes(exclude="number").columns]
        num_vars = [col for col in dataframe.select_dtypes("number").columns]

        cat_but_obj = [col for col in obj_vars if dataframe[col].nunique() <= cat_obj_th]
        cat_but_num = [col for col in num_vars if dataframe[col].nunique() <= cat_num_th]

        obj_vars = [col for col in obj_vars if col not in cat_but_obj]
        num_vars = [col for col in num_vars if col not in cat_but_num]
        cat_vars = cat_but_obj + cat_but_num

        if summary:
            print(" SUMMARY ".center(60, "*"))
            print(f"# of object variables: {len(obj_vars)}")
            print(obj_vars)
            print(f"# of numeric variables: {len(num_vars)}")
            print(num_vars)
            print(f"# of categoric variables: {len(cat_vars)}")
            print(cat_vars)

        return obj_vars, num_vars, cat_vars

    def lof_handler(self,dataframe:pd.DataFrame, num_cols:list ,lof_th = -2):
        """
        Local Outlier Factor handler. Detects outliers and drops them based on lof_th using
        LOF algorithm which is an unsupervised outlier detection method.

        Parameters
        ----------
        dataframe: pd.DataFrame
        num_cols: list
            Numeric variables list of the dataframe to be used in LOF algorithm.
        lof_th: int, default=-2
            Threshold of negative outlier factors. The more negative the value,
            the more likely the observation is an outlier. i.e. an observation having -5
            n.o.f is probably more anomalous than an observation having -2 n.o.f.
        """
        lof = LocalOutlierFactor()
        numerics = dataframe[num_cols]
        lof.fit(numerics)
        negatives = lof.negative_outlier_factor_

        lof_indexes = dataframe[negatives <= lof_th].index
        dataframe.drop(index=lof_indexes,inplace=True)

    def churn_prep(self,df: pd.DataFrame):
        # Feature Extraction
        df["CREDIT_RANGE"] = pd.cut(df["creditscore"],
                                        bins=[299, 579, 669, 739, 799, 850],
                                        labels=["poor", "fair", "good", "verygood", "excellent"])

        df["IS_BALANCE_0"] = np.where(df["balance"] == 0, 1, 0)

        df["SALARY_RANGE"] = pd.qcut(df["estimatedsalary"],
                                         q=5,
                                         labels=["low", "belowaverage", "average", "highaverage", "high"])

        df["AGE_CAT"] = pd.cut(df["age"],
                                bins=[0, 29, 49, 64, 150],
                                labels=[1, 2, 3, 4]).astype("int")

        df["AGE_TENURE"] = df["age"] * df["tenure"]
        df["TENURE_PRODUCTS"] = df["tenure"] * df["numofproducts"]

        # Grabbing columns with new variables
        obj, num, cat = self.grabbing_vars(df, cat_num_th=3)

        # Outlier Handling using LocalOutlierFactor
        self.lof_handler(df,num)

        # Binary Encoding
        bin_cols = [col for col in cat if df[col].nunique() == 2 and df[col].dtype == "object"]
        binencoder = LabelEncoder()
        for col in bin_cols:
            df[col] = binencoder.fit_transform(df[col])

        # One Hot Encoding
        ohe_cols = [col for col in cat if df[col].nunique() > 2 and df[col].dtype in ["object", "category"]]
        finaldf = pd.get_dummies(df, columns=ohe_cols, dtype="int", drop_first=True)

        # Separate target and independent dataframes
        X = finaldf.drop("exited", axis=1)
        Y = finaldf["exited"]

        # Scaling numerical variables using StandardScaler
        scaler = StandardScaler()
        X[num] = scaler.fit_transform(X[num])

        return X, Y


class ModelUtil:
    """
    Helper functions for observing model performance.
    """
    def feature_importance_plot(self, model, independent: pd.DataFrame, count=8):
        """
        Plots feature importances of the fitted model.

        Parameters
        ----------
        model:
            Fitted ML model for which features will be observed according to their importances
        independent: pd.DataFrame
            The dataframe containing independent variables only
        count: int, default=8
            Number of features will be plotted by descending order of their importances
        """
        feats = independent.columns
        if type(model).__name__ == "LogisticRegression":
            weights = np.abs(model.coef_[0])
        else:
            weights = model.feature_importances_
        feat_imp = pd.DataFrame({"Feature": feats,
                                 "Weight": weights})
        feat_imp.sort_values(by="Weight", ascending=False, inplace=True)
        sns.barplot(feat_imp[:count], x="Weight", y="Feature", hue="Feature")
        plt.title(f"Feature Importance of {type(model).__name__}")
        plt.show(block=True)

    def model_metrics_summary(self, model, x:pd.DataFrame, y:pd.DataFrame, cv=5, metrics=["accuracy","f1","roc_auc"]):
        """
        Plots the model's metric scores by given metrics.

        Parameters
        ----------
        model:
            ML Model (fitted or not) will be validated
        x: pd.DataFrame
            Dataframe containing independent variables only
        y: pd.DataFrame
            Dataframe containing a dependent variable only
        cv: int, default=5
            Number of cross validation
        metrics: list, default=["accuracy","f1","roc_auc"]
        """
        scores = []
        cv_results = cross_validate(model, x, y, cv=cv, scoring=metrics)
        for metric in metrics:
            scores.append(cv_results[f"test_{metric}"].mean())

        sum_metric = pd.DataFrame({"Metric": metrics,
                                   "Score": scores})
        ax = sns.barplot(data=sum_metric, y="Metric", x="Score", hue="Metric")
        for i in range(len(metrics)):
            ax.bar_label(ax.containers[i])
        plt.title(f"{type(model).__name__} Metric Scores")
        plt.show(block=True)





