import pandas as pd
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy.stats import chi2_contingency, ks_2samp


def hypothesis_tests(data):
    # Chi-squared test for Y against X1, X2, ..., X23
    for col in data.columns[:-1]:  # All columns except the last (Y)
        contingency_table = pd.crosstab(data['Y'], data[col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        if p < 0.05:
            print(f"Chi-squared test for {col}: Null hypothesis rejected (distributions are significantly different)")

    # Kolmogorov-Smirnov test
    stat, p = ks_2samp(data['Y'], data['Predictions'])
    if p < 0.05:
        print("Kolmogorov-Smirnov test: Null hypothesis rejected (distributions are significantly different)")


def main():
    data = pd.read_csv("credit_train.csv")
    X = data.drop("Y", axis=1)
    Y = data["Y"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)
    
    model = LogisticRegression().fit(X_train, Y_train)
    
    Y_hat_train = model.predict(X_train)
    Y_hat_test = model.predict(X_test)

    # Save predictions for analysis
    predictions_df = pd.DataFrame({'Predictions': Y_hat_test, 'Actual': Y_test})
    predictions_df.to_csv('predictions.csv', index=False)

    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)
    
    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)
    
    print(f"Train f1: {f1_train}")
    print(f"Test f1: {f1_test}")
    print(f"Train accuracy: {accuracy_train}")
    print(f"Test accuracy: {accuracy_test}")
    
    # Saving our model
    with open("./models/model.pkl", "wb") as file:
        pickle.dump(model, file)

if __name__ == "__main__":
    main()