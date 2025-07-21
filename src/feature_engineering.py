import pandas as pd
import numpy as np

def add_time_features(df):
    """
    Add hour_of_day, day_of_week, and time_since_signup features to the DataFrame.
    Args:
        df (pd.DataFrame): Input DataFrame with 'signup_time' and 'purchase_time' as datetime.
    Returns:
        pd.DataFrame: DataFrame with new time-based features.
    """
    df = df.copy()
    df['hour_of_day'] = df['purchase_time'].dt.hour
    df['day_of_week'] = df['purchase_time'].dt.dayofweek
    df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600.0  # in hours
    return df

def add_transaction_frequency(df):
    """
    Add transaction frequency and velocity features per user and per device.
    Args:
        df (pd.DataFrame): Input DataFrame with 'user_id', 'device_id', and 'purchase_time'.
    Returns:
        pd.DataFrame: DataFrame with new frequency and velocity features.
    """
    df = df.copy()
    # Transaction count per user
    user_tx_count = df.groupby('user_id')['purchase_time'].transform('count')
    df['user_transaction_count'] = user_tx_count
    # Transaction count per device
    device_tx_count = df.groupby('device_id')['purchase_time'].transform('count')
    df['device_transaction_count'] = device_tx_count
    # Transaction velocity (transactions per day per user)
    df['user_first_purchase'] = df.groupby('user_id')['purchase_time'].transform('min')
    df['user_days_since_first'] = (df['purchase_time'] - df['user_first_purchase']).dt.total_seconds() / (3600*24)
    df['user_transaction_velocity'] = df['user_transaction_count'] / (df['user_days_since_first'] + 1)
    df.drop(['user_first_purchase', 'user_days_since_first'], axis=1, inplace=True)
    return df

def ip_to_int(ip_str):
    """
    Convert an IPv4 address string to its integer representation.
    Args:
        ip_str (str): IPv4 address as string.
    Returns:
        int: Integer representation of the IP address.
    """
    try:
        parts = [int(x) for x in ip_str.split('.')]
        return (parts[0] << 24) + (parts[1] << 16) + (parts[2] << 8) + parts[3]
    except Exception:
        return np.nan

def add_ip_integer_column(df, ip_col='ip_address'):
    """
    Add a column with integer representation of IP addresses.
    Args:
        df (pd.DataFrame): DataFrame with IP address column.
        ip_col (str): Name of the IP address column.
    Returns:
        pd.DataFrame: DataFrame with new 'ip_address_int' column.
    """
    df = df.copy()
    df['ip_address_int'] = df[ip_col].astype(str).apply(ip_to_int)
    return df

def merge_with_country(df, ip_country_df):
    """
    Merge transaction data with IP-to-country mapping using integer IP addresses.
    Args:
        df (pd.DataFrame): Transaction DataFrame with 'ip_address_int'.
        ip_country_df (pd.DataFrame): IP-to-country DataFrame with 'lower_bound_ip_address' and 'upper_bound_ip_address'.
    Returns:
        pd.DataFrame: Merged DataFrame with 'country' column (NaN filled as 'Unknown').
    """
    df = df.copy()
    ip_country_df = ip_country_df.copy()
    # Create IntervalIndex for IP ranges
    intervals = pd.IntervalIndex.from_arrays(
        ip_country_df['lower_bound_ip_address'],
        ip_country_df['upper_bound_ip_address'],
        closed='both'
    )
    # Map each ip_address_int to a country
    def find_country(ip):
        if pd.isna(ip):
            return np.nan
        idx = intervals.get_indexer([ip])[0]
        if idx == -1:
            return np.nan
        return ip_country_df.iloc[idx]['country']
    df['country'] = df['ip_address_int'].apply(find_country)
    # Fill NaN with 'Unknown'
    df['country'] = df['country'].fillna('Unknown')
    return df