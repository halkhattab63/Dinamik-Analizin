import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

def load_dataset(path):
    dataset = pd.read_csv(path)
    X = dataset.iloc[:, [2, 3]].values  # اختر ميزتين فقط
    y = dataset.iloc[:, 4].values       # اختر العمود الهدف
    return X, y

def preprocess_data(X, y, bins=5, test_size=0.25, random_state=0):
    kbd = KBinsDiscretizer(n_bins=bins, encode='ordinal', strategy='uniform')
    y_binned = kbd.fit_transform(y.reshape(-1, 1)).astype(int).ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y_binned, test_size=test_size, 
                                                        random_state=random_state, stratify=y_binned)

    sc_X = StandardScaler()
    X_train = sc_X.fit_transform(X_train)
    X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test, y_binned
