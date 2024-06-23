import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_handler(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    scaler = StandardScaler()
    scaler.fit(train_df[['temp']])

    train_scaled_data = scaler.transform(train_df[['temp']])
    test_scaled_data = scaler.transform(test_df[['temp']])

    train_df['temp'] = train_scaled_data
    train_df.to_csv('train/train_result.csv', index=False)

    test_df['temp'] = test_scaled_data
    test_df.to_csv('test/test_result.csv', index=False)

print('start: preprocess data')
preprocess_handler(train_path='train/train_data.csv', test_path='test/test_data.csv')
print('finish: preprocess data')
