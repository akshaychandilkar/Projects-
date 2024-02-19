import math
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def MarvellousTitanicLogistic():

    # Step1: Load Data
    titanic_data = pd.read_csv('MarvellousTitanicDataset.csv')

    print("First 5 entries from loaded dataset")
    print(titanic_data.head())

    print("Number of passangers are "+str(len(titanic_data)))

    #Step 2: Analyze data
    print("Visualisation: Survived and non survied passangers")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target).set_title("Marvellous Infosystems :Survived and non survied passangers")
    show()

    print("Visualisation: Survived and non survied passangers based on Gender")
    figure()
    target="Survived"

    countplot(data=titanic_data,x=target, hue="Sex").set_title("Marvellous Infosystems: Survived and non survied passangers based on Gender")
    show()

    print("Visualisation: Survived and non survied passangers based on the Passanger class")
    figure()
    target = "Survived"

    countplot(data=titanic_data,x=target, hue="Pclass").set_title("Marvellous Infosystems: Survived and non survied passangers based on the Passanger class")
    show()

    print("Visualisation: Survived and non survied passangers based on Age")
    figure()
    titanic_data["Age"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers based on Age")
    show()

    print("Visualisation: Survived and non survied passangers based on the Fare")
    figure()
    titanic_data["Fare"].plot.hist().set_title("Marvellous Infosystems: Survived and non survied passangers based on Fare")
    show()

    # Step 3: Data Cleaning
    titanic_data.drop("zero", axis=1, inplace=True)
    x = titanic_data.drop("Survived", axis=1)

    x = x.rename(columns=lambda col: str(col))
    y = titanic_data["Survived"]

    print("First 5 entries from loaded dataset after removing zero column")
    print(titanic_data.head(5))

    print("Values of Sex column")
    print(pd.get_dummies(titanic_data["Sex"]))

    print("Values of Sex column after removing one field")
    Sex = pd.get_dummies (titanic_data["Sex"], drop_first = True)
    print(Sex.head(5))

    print("Values of Plass column after removing one field")
    Pclass= pd.get_dummies (titanic_data["Pclass"], drop_first = True)
    print(Pclass.head(5))

    print("Values of data set after concatenating new columns")
    titanic_data= pd.concat([titanic_data,Sex, Pclass],axis=1)
    print(titanic_data.head(5))

    print("Values of data set after removing irrelevent columns")
    titanic_data.drop(["Sex", "sibsp", "Parch", "Embarked"], axis= 1, inplace=True)
    print(titanic_data.head(5))

    # Step 4: Handle Missing Values
    imputer = SimpleImputer(strategy='mean')
    imputer.fit(x)
    x_imputed = imputer.transform(x)

    # Step4: Data Training
    xtrain, xtest, ytrain, ytest, = train_test_split(x_imputed, y, test_size=0.5, random_state=42)

    logmodel = LogisticRegression(max_iter=1500)

    # Now it should work without the TypeError
    logmodel.fit(xtrain, ytrain)

    # Step 4: Data Testing
    prediction = logmodel.predict(xtest)

    #Step 5: Calculate Accuracy
    print("Classification report of Logistic Regression is: ")
    print(classification_report(ytest,prediction))

    print("Confusion Matrix of Logistic Regression is: ")
    print(confusion_matrix(ytest,prediction))

    print("Accuracy of Logistic Regression is: ")
    print(accuracy_score(ytest,prediction))

def main():
    print("----Marvellous Infosystems by Piyush Khairnar----")

    print("Suervised Machine Learning")

    print("Logistic Regreesion on Titanic data set")

    MarvellousTitanicLogistic()

if __name__ == "__main__":
    main()