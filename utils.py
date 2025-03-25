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
    df.columns = df.columns.str.strip()

    # Define required columns
    required_columns = {'Material Number', 'Pstng Date', 'Quantity'}
    
    # Check if all required columns are present
    if not required_columns.issubset(df.columns):
        missing = required_columns - set(df.columns)
        raise ValueError(f"Missing columns in the data: {', '.join(missing)}")
    
    
    # Convert posting date to datetime
    df['Pstng Date'] = pd.to_datetime(df['Pstng Date'])
    
    # Extract the ISO week number
    df['Week'] = df['Pstng Date'].dt.isocalendar().week

    # Take absolute of Quantity before grouping
    df['Quantity'] = df['Quantity'].abs()
    
    # Group by Material Number and Week, summing the quantity
    grouped = df.groupby(['Material Number', 'Week'])['Quantity'].sum().reset_index()
    
    # Pivot the table to have weeks as columns
    pivot_df = grouped.pivot(index='Material Number', columns='Week', values='Quantity').fillna(0)
    
    # Rename the week columns to WW1_Consumption, WW2_Consumption, ..., WW52_Consumption
    pivot_df.columns = [f'WW{week}_Consumption' for week in pivot_df.columns]
    
    # Reset index to bring 'Material Number' back as a column
    result_df = pivot_df.reset_index()
    
    return result_df