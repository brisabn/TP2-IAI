import pandas as pd
import random
from sklearn.preprocessing import StandardScaler

def get_attribute_names(file_path):
    df = pd.read_csv(file_path)
    return list(df.columns)

def load_and_split_data(train_file, test_file, target_column, train_size=0.8):
    # Gerar um valor aleatório para random_state entre 1 e 100
    random_state = random.randint(1, 100)
    # print("Random State:", random_state)

    # lendo o arquivo csv
    df_train = pd.read_csv(train_file)
    df_test = pd.read_csv(test_file)

    # shuffle o arquivo para selecionar aleatoriamente
    df_train = df_train.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df_test = df_test.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # separa o target
    y_train = df_train[target_column]
    y_test = df_test[target_column]

    # temove o target
    x_train = df_train.drop(target_column, axis=1)
    x_test = df_test.drop(target_column, axis=1)

    # normalização usando StandardScaler
    scaler = StandardScaler()
    x_train_normalized = scaler.fit_transform(x_train)
    x_test_normalized = scaler.transform(x_test)

    # tamannho do datset
    train_size = int(train_size * len(df_train))
    test_size = len(df_train) - train_size

    # separa o dataset
    x_train_normalized = x_train_normalized[:train_size]
    y_train = y_train[:train_size]
    x_test_normalized = x_test_normalized[:test_size]
    y_test = y_test[:test_size]

    return x_train_normalized, y_train, x_test_normalized, y_test
