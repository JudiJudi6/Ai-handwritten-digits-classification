import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Ścieżki do plików CSV
train_dir = './archive/mnist_train.csv'
test_dir = './archive/mnist_test.csv'

# Wczytanie danych treningowych i testowych z plików CSV
train_data = pd.read_csv(train_dir)
test_data = pd.read_csv(test_dir)

# Podział danych na cechy i etykiety
X_train = train_data.drop(columns=['label']).values  # cechy (usuń kolumnę 'label')
y_train = train_data['label'].values                 # etykiety

X_test = test_data.drop(columns=['label']).values   # cechy
y_test = test_data['label'].values                  # etykiety

# Normalizacja danych za pomocą Min-Max Scaler
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Sprawdzenie wymiarów po normalizacji (opcjonalne)
print("Rozmiar X_train po normalizacji:", X_train_normalized.shape)
print("Rozmiar X_test po normalizacji:", X_test_normalized.shape)

# Zapisanie znormalizowanych danych do nowych plików CSV (opcjonalne)
train_normalized_dir = './archive/mnist_train_normalized.csv'
test_normalized_dir = './archive/mnist_test_normalized.csv'

train_normalized_data = pd.DataFrame(X_train_normalized)
train_normalized_data.insert(0, 'label', y_train)  # dodaj etykiety
train_normalized_data.to_csv(train_normalized_dir, index=False)

test_normalized_data = pd.DataFrame(X_test_normalized)
test_normalized_data.insert(0, 'label', y_test)  # dodaj etykiety
test_normalized_data.to_csv(test_normalized_dir, index=False)

print("Znormalizowane dane zostały zapisane do plików CSV.")
