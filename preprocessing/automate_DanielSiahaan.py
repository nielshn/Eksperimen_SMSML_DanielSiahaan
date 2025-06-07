import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os


def load_and_merge_datasets(red_path, white_path):
    red_df = pd.read_csv(red_path, sep=';')
    white_df = pd.read_csv(white_path, sep=';')
    red_df['wine_type'] = 'red'
    white_df['wine_type'] = 'white'
    combined_df = pd.concat([red_df, white_df], axis=0).reset_index(drop=True)
    return combined_df


def preprocess_data(df):
    le = LabelEncoder()
    df['wine_type'] = le.fit_transform(df['wine_type'])
    X = df.drop('quality', axis=1)
    y = df['quality']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    X_scaled_df['quality'] = y
    return X_scaled_df


def save_preprocessed_data(df, output_path):
    dir_name = os.path.dirname(output_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    df.to_csv(output_path, index=False)


def main():
    red_path = "winequality-red.csv"
    white_path = "winequality-white.csv"
    output_path = "preprocessing/winequality_preprocessed.csv"

    print("Loading datasets...")
    raw_df = load_and_merge_datasets(red_path, white_path)

    print("Preprocessing data...")
    processed_df = preprocess_data(raw_df)

    print(f"Saving preprocessed dataset to {output_path}...")
    save_preprocessed_data(processed_df, output_path)

    print("Done! Dataset ready for training")


if __name__ == "__main__":
    main()
