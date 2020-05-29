import pandas as pd
from split import split_func
from scaler import scaling
from results import model
from metrics import cf_matrix

data = pd.read_csv("data.csv")

# We will bw using age, gender and estimated salary to predict the click on ad
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values

X_train, X_test, y_train, y_test = split_func(X, y)
X_train, X_test = scaling(X_train, X_test)

y_pred = model(X_train, y_train, X_test, y_test)

cm = cf_matrix(y_test, y_pred)
print(cm)



