import pandas as pd
import numpy as np

def load_data_consumption(file):
    """
    Loads data from an Excel file and selects specific columns,
    performing necessary data type conversions and cleaning.

    Args:
        file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded Excel file.

    Returns:
        pandas.DataFrame: The DataFrame with selected columns and processed data.
    """
    df = pd.read_excel(file)
    # Strip leading and trailing spaces from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    # Select specific columns
    selected_cols = ['Material Group', 'Material Number', 'Pstng Date', 'Quantity',
                     'BUn', 'Plant', 'Site', 'Batch', 'SLED/BBD', 'Vendor Number']
    df = df[selected_cols]

    # Convert 'Pstng Date' to datetime
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])

    # Convert 'SLED/BBD' to datetime, handling errors and filling NaT
    df['SLED/BBD'] = pd.to_datetime(df['SLED/BBD'], errors='coerce')
    df['SLED/BBD'] = df['SLED/BBD'].fillna(pd.to_datetime('2100-01-01'))

    # Convert negative consumption values to positive
    df['Quantity'] = df['Quantity'].abs()

    return df


def load_data_GR(file):
    """
    Loads data from an Excel file and selects specific columns,
    performing necessary data type conversions and cleaning.

    Args:
        file (streamlit.runtime.uploaded_file_manager.UploadedFile):
            The uploaded Excel file.

    Returns:
        pandas.DataFrame: The DataFrame with selected columns and processed data.
    """
    df = pd.read_excel(file)
    # Strip leading and trailing spaces from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    # Convert 'Pstng Date' to datetime
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])

    # Convert 'SLED/BBD' to datetime, handling errors and filling NaT
    df['SLED/BBD'] = pd.to_datetime(df['SLED/BBD'], errors='coerce')
    df['SLED/BBD'] = df['SLED/BBD'].fillna(pd.to_datetime('2100-01-01'))

    # Convert negative consumption values to positive
    df['Quantity'] = df['Quantity'].abs()

    return df


def load_data(file):
    df = pd.read_excel(file)
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])

    # Strip leading and trailing spaces from all string columns
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    df.columns = df.columns.str.strip()

    return df

def load_forecast_consumption_data(file):
    df = pd.read_excel(file)
    return df