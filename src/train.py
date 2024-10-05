import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import chi2_contingency, ks_2samp

def hypothesis_tests(data):
    """
    Realiza pruebas de hipótesis en el conjunto de datos.
    Se llevan a cabo pruebas Chi-cuadrado y Kolmogorov-Smirnov.
    
    Args:
        data (pd.DataFrame): DataFrame que contiene las características y la variable objetivo.
    """
    # Chi-squared test for Y against each feature
    for col in data.columns[:-1]:  # All columns except the last (Y)
        contingency_table = pd.crosstab(data['Y'], data[col])
        chi2, p, _, _ = chi2_contingency(contingency_table)
        if p < 0.05:
            print(f"Chi-squared test for {col}: Null hypothesis rejected (p-value: {p:.4f})")
        else:
            print(f"Chi-squared test for {col}: Null hypothesis accepted (p-value: {p:.4f})")

    # Kolmogorov-Smirnov test
    stat, p = ks_2samp(data['Y'], data['Predictions'])
    if p < 0.05:
        print("Kolmogorov-Smirnov test: Null hypothesis rejected (p-value: {p:.4f})")
    else:
        print("Kolmogorov-Smirnov test: Null hypothesis accepted (p-value: {p:.4f})")

def main():
    data = pd.read_csv("./data/credit_train.csv")
    X = data.drop("Y", axis=1)
    Y = data["Y"]
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1234)
    
    # Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Entrenamiento del modelo
    model = LogisticRegression(solver='liblinear', random_state=1234)
    model.fit(X_train_scaled, Y_train)
    
    Y_hat_train = model.predict(X_train_scaled)
    Y_hat_test = model.predict(X_test_scaled)

    # Guardar predicciones para análisis
    predictions_df = pd.DataFrame({'Predictions': Y_hat_test, 'Actual': Y_test})
    predictions_df.to_csv('predictions.csv', index=False)

    # Métricas de rendimiento
    f1_train = f1_score(Y_train, Y_hat_train)
    f1_test = f1_score(Y_test, Y_hat_test)
    
    accuracy_train = accuracy_score(Y_train, Y_hat_train)
    accuracy_test = accuracy_score(Y_test, Y_hat_test)
    
    print(f"Train f1: {f1_train:.4f}")
    print(f"Test f1: {f1_test:.4f}")
    print(f"Train accuracy: {accuracy_train:.4f}")
    print(f"Test accuracy: {accuracy_test:.4f}")

    # Guardar el modelo y el escalador
    with open("./models/model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    with open("./models/scaler.pkl", "wb") as file:
        pickle.dump(scaler, file)

    # Crear un nuevo DataFrame para las predicciones y combinarlas
    test_predictions_df = pd.DataFrame(X_test, columns=X.columns)  # DataFrame de X_test
    test_predictions_df['Predictions'] = Y_hat_test  # Agregar las predicciones
    test_predictions_df['Actual'] = Y_test.reset_index(drop=True)  # Agregar los valores reales

    # Guardar las nuevas predicciones
    test_predictions_df.to_csv('test_predictions.csv', index=False)

    # Realizar pruebas de hipótesis
    data['Predictions'] = model.predict(scaler.transform(data.drop("Y", axis=1)))  # Predicciones en el conjunto completo
    hypothesis_tests(data)

if __name__ == "__main__":
    main()