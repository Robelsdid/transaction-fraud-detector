import pandas as pd

def load_csv_data(filepath):
    """
    Load a CSV file into a pandas DataFrame.
    Args:
        filepath (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_csv(filepath)

def check_missing_values(df):
    """
    Print the number of missing values per column in the DataFrame.
    Args:
        df (pd.DataFrame): DataFrame to check.
    Returns:
        pd.Series: Series with missing value counts per column.
    """
    missing = df.isnull().sum()
    print(missing[missing > 0])
    return missing

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with duplicates removed.
    """
    return df.drop_duplicates()

def correct_fraud_data_types(df):
    """
    Correct data types in Fraud_Data.csv DataFrame.
    Converts signup_time and purchase_time to datetime, age to int, purchase_value to float.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with corrected types.
    """
    df = df.copy()
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    df['age'] = df['age'].astype(int)
    df['purchase_value'] = df['purchase_value'].astype(float)
    return df

def correct_ip_country_data_types(df):
    """
    Correct data types in IpAddress_to_Country.csv DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with corrected types.
    """
    df = df.copy()
    df['lower_bound_ip_address'] = df['lower_bound_ip_address'].astype(str)
    df['upper_bound_ip_address'] = df['upper_bound_ip_address'].astype(str)
    df['country'] = df['country'].astype(str)
    return df
