# Tasks
# 1: Analyze the dataset to estimate property prices as a regression problem
"""here we will predict a continuous number rather than a certain class, thats why it is a regression problem
in our case, the regression problem is: predicting the price of a property based on its features
Target variable: inm_price column in the dataset
Input features: the attributes of the property (floor, size etc.),
histografic info of the area( old price, annual variation etc.) and the demographic info
"""
# We first need to analyze the dataset and detect outliers, the distribution of the featues etc.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def dataset_analysis():
    # Loading dataset
    # dataset = pd.read_excel("session_7_dataset.xlsx")
    dataset = pd.read_csv("session_7_dataset.csv")
    """#converting to csv, I did this because reading csv is easier than reading excel files
    dataset.to_csv("session_7_dataset.csv", index=False)
    """
    # how does dataset look like
    print(dataset.shape)  # number of rows and columns
    # some statistics about the dataset
    print("*******************Summary statistics:********************")
    print(dataset.describe())  # mean, std, min, max, quartiles
    # how many null value exists for each column, I realized some fields have a lot of null values
    # but for example barrio needs to be filled because it is important for price prediction
    print("*********************Null Counts:****************************")
    print(dataset.isnull().sum())
    # price distribution over the dataset
    plt.figure(figsize=(8, 5))
    sns.histplot(dataset["inm_price"], bins=50, kde=True)
    plt.title("Distribution of Property Prices")
    plt.show()
    # correlation matrix but I excluded barrio and district bc they need to be mapped
    plt.figure(figsize=(16, 12))
    corr = dataset.select_dtypes(include=["number"]).corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm", cbar=True, square=True)
    plt.title("Correlation Heatmap", fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()
    plt.show()

    # relationship between apartment size and price
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x="inm_size", y="inm_price", data=dataset)
    plt.title("Property Size vs Price")
    plt.show()

    # average price per district, I used this during integer mapping
    plt.figure(figsize=(14, 8))
    avg_price_per_district = dataset.groupby("inm_distrito")["inm_price"].mean().sort_values(ascending=False)
    print("Average property price per district (highest to lowest):\n")
    print(avg_price_per_district)
    avg_price_per_district.plot(kind="bar")
    plt.title("Average Price by District", fontsize=14)
    plt.ylabel("Average Price", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)  # rotate district names
    plt.tight_layout()  # prevent labels from being cut off
    plt.show()


"""
Outcomes that I obtained from dataset analysis:
From correlation table: Size (inm_size), historical price(his_price) and proportion of university graduates or students (dem_PropConEstudiosUniversitarios)
is positively related with price, and unemployment rate (dem_TasaDeParo), proportion without university studies (dem_PropSinEstudiosUniversitarios) and
proportion without any studies (dem_PropSinEstudios) are negatively related with price.
From district-price table: the top 8 districts which has the highest price are:
Moncloa - Aravaca, Salamanca, Chamartín, Hortaleza, Chamberí, Retiro, Fuencarral - El Pardo, Centro
I decided to map the districts and barrios to integers based on their average price, the most expensive district would be the highest number
From property size vs price graph: property size is positively related with price which was seen from correlation graph too.
From property price distribution graph: I saw that most of the properties have low price, the distribution is not like a normal distribution, its skewed.
"""


def apply_linear_regression():
    # reading the dataset
    dataset = pd.read_csv("session_7_dataset.csv")
    # unnamed is dropped bcause it is just an index column
    dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")
    # I decided to map the districts to numbers, the most expensive district would be the highest number
    # cheapest = 1, most expensive = N, null ones are 0
    district_ranks = dataset.groupby("inm_distrito")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_distrito_encoded"] = dataset["inm_distrito"].map(district_ranks).fillna(0)
    # cheapest = 1, most expensive = N, null ones are 0
    barrio_ranks = dataset.groupby("inm_barrio")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_barrio_encoded"] = dataset["inm_barrio"].map(barrio_ranks).fillna(0)
    # I selected these features based on the dataset analysis that I made earlier
    selected_features = [
        "inm_size",
        "his_price",
        "dem_PropConEstudiosUniversitarios",
        "dem_TasaDeParo",
        "dem_PropSinEstudiosUniversitarios",
        "dem_PropSinEstudios",
        "inm_distrito_encoded",
        "inm_barrio_encoded"
    ]
    # Feature matrix and target variable
    X = dataset[selected_features]
    y = dataset["inm_price"]
    # filling null values with the mean of the column
    X = X.fillna(X.mean())
    # splitting the dataset into test and train, only 20% of the data is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # training the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    # testing the model
    y_pred = model.predict(X_test)
    # accuracy metrics shown in class: rse, r^2
    n = len(y_test)
    # Number of features
    p = X_test.shape[1]
    # Residuals
    residuals = y_test - y_pred
    # Residual Standard Error
    rse = (sum(residuals ** 2) / (n - p - 1)) ** 0.5
    # R^2 score
    r2 = r2_score(y_test, y_pred)
    print("Model Evaluation:")
    print("-----------------")
    print(f"Residual Standard Error (RSE): {rse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    # plotting the predicted vs actual prices
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title("Predicted vs Actual Property Prices")
    plt.show()


def apply_polynomial_regression():
    # loading the data
    dataset = pd.read_csv("session_7_dataset.csv")
    # unnamed is dropped bcause it is just an index column
    dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")
    # I decided to map the districts to numbers, the most expensive district would be the highest number
    # cheapest = 1, most expensive = N, null ones are 0
    district_ranks = dataset.groupby("inm_distrito")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_distrito_encoded"] = dataset["inm_distrito"].map(district_ranks).fillna(0)
    # cheapest = 1, most expensive = N, null ones are 0
    barrio_ranks = dataset.groupby("inm_barrio")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_barrio_encoded"] = dataset["inm_barrio"].map(barrio_ranks).fillna(0)
    # I selected these features based on the dataset analysis that I made earlier
    selected_features = [
        "inm_size",
        "his_price",
        "dem_PropConEstudiosUniversitarios",
        "dem_TasaDeParo",
        "dem_PropSinEstudiosUniversitarios",
        "dem_PropSinEstudios",
        "inm_distrito_encoded",
        "inm_barrio_encoded"
    ]
    # feature matrix and target variable
    X = dataset[selected_features]
    y = dataset["inm_price"]
    # filling null values with the mean of the column
    X = X.fillna(X.mean())
    # splitting the dataset into test and train, only 20% of the data is used for testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # training the model
    # polynomial feature transformation
    poly = PolynomialFeatures(degree=3, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # fitting linear Regression on polynomial features
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    # predicting
    y_pred = model.predict(X_test_poly)

    # evaluating with rse and r^2
    n = len(y_test)
    p = X_test_poly.shape[1]
    residuals = y_test - y_pred
    rse = (sum(residuals ** 2) / (n - p - 1)) ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Polynomial Regression Evaluation:")
    print("--------------------------------")
    print(f"Degree: 3")
    print(f"Residual Standard Error (RSE): {rse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    # predicted vs actual prices plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", lw=2)
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.title(f"Polynomial Regression (degree=3)\nPredicted vs Actual Prices")
    plt.show()


# 2: Analyze data and redefine the problem to create a classification one
# Algorithms that should be used: Perceptron learning, Logistic regression, Generative models (LDA, QDA), KNN
def apply_classification_models():
    dataset = pd.read_csv("session_7_dataset.csv")
    dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")
    # previously defined district and barrio encoding
    district_ranks = dataset.groupby("inm_distrito")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_distrito_encoded"] = dataset["inm_distrito"].map(district_ranks).fillna(0)
    barrio_ranks = dataset.groupby("inm_barrio")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_barrio_encoded"] = dataset["inm_barrio"].map(barrio_ranks).fillna(0)
    # Here, I classified the prices into 3 classes: low, medium, high
    """
    # and I used mean and standard deviation to define the classes, but it didnt give good results so I abandoned it
    mean_price = dataset['inm_price'].mean()
    std_price = dataset['inm_price'].std()
    def price_to_class(price):
        if price < mean_price - std_price:
            return 0  # Low
        elif price > mean_price + std_price:
            return 2  # High
        else:
            return 1  # Medium
    # creating the price class column
    dataset['price_class'] = dataset['inm_price'].apply(price_to_class)"""
    dataset['price_class'] = pd.qcut(dataset['inm_price'], q=3, labels=[0, 1, 2])
    # checking class distribution
    print("Class distribution:")
    print(dataset['price_class'].value_counts())
    # previously selected features
    features = [
        "inm_size",
        "his_price",
        "dem_PropConEstudiosUniversitarios",
        "dem_TasaDeParo",
        "dem_PropSinEstudiosUniversitarios",
        "dem_PropSinEstudios",
        "inm_distrito_encoded",
        "inm_barrio_encoded"
    ]
    X = dataset[features].fillna(0)
    y = dataset['price_class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # importing all the models that we need
    models = {
        "Perceptron": Perceptron(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "LDA": LinearDiscriminantAnalysis(),
        "QDA": QuadraticDiscriminantAnalysis(),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


# 3: Apply other methods (make research for this) and compare their results
def apply_other_classifications():
    dataset = pd.read_csv("session_7_dataset.csv")
    dataset = dataset.drop(columns=["Unnamed: 0"], errors="ignore")
    # previously defined district and barrio encoding
    district_ranks = dataset.groupby("inm_distrito")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_distrito_encoded"] = dataset["inm_distrito"].map(district_ranks).fillna(0)
    barrio_ranks = dataset.groupby("inm_barrio")["inm_price"].mean().rank(ascending=True).astype(int)
    dataset["inm_barrio_encoded"] = dataset["inm_barrio"].map(barrio_ranks).fillna(0)

    # same price classification as before
    dataset['price_class'] = pd.qcut(dataset['inm_price'], q=3, labels=[0, 1, 2])
    print("Class distribution:")
    print(dataset['price_class'].value_counts())
    # previously selected features
    features = [
        "inm_size",
        "his_price",
        "dem_PropConEstudiosUniversitarios",
        "dem_TasaDeParo",
        "dem_PropSinEstudiosUniversitarios",
        "dem_PropSinEstudios",
        "inm_distrito_encoded",
        "inm_barrio_encoded"
    ]
    X = dataset[features].fillna(0)
    y = dataset['price_class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # importing models that I decided to use
    # I decided to use Decision Tree, Random Forest and SVM because I have heard about them in other courses
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM": SVC(kernel='rbf', C=1.0, random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n=== {name} ===")
        print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("Classification Report:\n", classification_report(y_test, y_pred))


def main():
    # part 1
    print("********************Part 1********************")
    dataset_analysis()
    apply_linear_regression()
    apply_polynomial_regression()
    # part 2
    print("********************Part 2********************")
    apply_classification_models()
    # part 3
    print("********************Part 3********************")
    apply_other_classifications()


if __name__ == "__main__":
    main()
