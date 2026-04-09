from sklearn.preprocessing import StandardScaler

def get_labels(df):
    y = df["dialect"].str[2].astype(int) - 1

    return y

def get_split(X, y, df, split):
    mask = df["split"] == split.upper()

    return X[mask], y[mask]

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
