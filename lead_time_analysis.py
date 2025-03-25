import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def process_dataframes(op_df, gr_df):
    """
    Finds matching rows in two DataFrames based on 'Material Number', 'Vendor Number',
    and 'Purchasing Document', concatenates 'Document Date' and 'Pstng Date',
    and identifies unmatched rows.

    Args:
        op_df (pd.DataFrame): DataFrame with 'Material Number', 'Purchasing Document',
                              'Vendor Number', and 'Document Date' columns.
        gr_df (pd.DataFrame): DataFrame with 'Material Number', 'Purchasing Document',
                              'Vendor Number', and 'Pstng Date' columns.

    Returns:
        tuple: A tuple containing three DataFrames:
               - matched_df: DataFrame with matched rows and concatenated dates.
               - unmatched_op_df: DataFrame with rows from op_df that have no match.
               - unmatched_gr_df: DataFrame with rows from gr_df that have no match.
    """

    # Merge DataFrames based on common columns
    merged_df = pd.merge(op_df, gr_df, on=['Material Number', 'Purchasing Document', 'Plant'],
                         how='outer', indicator=True)

    # Find matched rows
    matched_df = merged_df[merged_df['_merge'] == 'both'].copy()
    matched_df['Combined Date'] = matched_df['Document Date'].astype(str) + ' | ' + matched_df['Pstng Date'].astype(str)
    matched_df = matched_df.drop(['_merge'], axis=1)

    # Find unmatched rows
    unmatched_op_df = merged_df[merged_df['_merge'] == 'left_only'].drop(['_merge', 'Pstng Date'], axis=1)
    unmatched_gr_df = merged_df[merged_df['_merge'] == 'right_only'].drop(['_merge', 'Document Date'], axis=1)

    return matched_df, unmatched_op_df, unmatched_gr_df



def calculate_actual_lead_time(df):
  """
  Calculates the Actual Lead Time in days for a DataFrame.

  Args:
    df: Pandas DataFrame with 'Document Date' and 'Pstng Date' columns.

  Returns:
    Pandas DataFrame with an added 'Actual Lead Time' column.
  """

  # Ensure the date columns are in datetime format
  df['Document Date'] = pd.to_datetime(df['Document Date'], errors='coerce')
  df['Pstng Date'] = pd.to_datetime(df['Pstng Date'], errors='coerce')

  # Calculate the difference in days
  df['Actual Lead Time'] = (df['Pstng Date'] - df['Document Date']).dt.days

  return df

def calculate_lead_time_summary(df):
    """
    Calculates the maximum and minimum lead times for each material number from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: A DataFrame containing the plant, site, material group,
                          material number, supplier, maximum lead time, and minimum lead time.
                          Returns None if an error occurs during file loading.
    """

    lead_time_cols = [col for col in df.columns if 'Lead Time WW' in col]
    grouped = df.groupby('Material Number')
    result = []

    for material_num, group in grouped:
        plant = group['Plant'].iloc[0]
        site = group['Site'].iloc[0]
        material_group = group['Material Group'].iloc[0]
        supplier = group['Supplier'].iloc[0]

        lead_times = group[lead_time_cols].values.flatten()
        lead_times = lead_times[~pd.isnull(lead_times)]
        if len(lead_times) > 0: #prevent error if lead_times is empty
            max_lead_time = lead_times.max()
            min_lead_time = lead_times.min()
        else:
            max_lead_time = None
            min_lead_time = None

        result.append({
            'Plant': plant,
            'Site': site,
            'Material Group': material_group,
            'Material Number': material_num,
            'Supplier': supplier,
            'Max Lead Time': max_lead_time,
            'Min Lead Time': min_lead_time
        })

    final_df = pd.DataFrame(result)
    return final_df

def calculate_lead_time_differences(final_df, calculated_df):
    """
    Calculates the lead time differences between final and actual lead times.

    Args:
        final_df (pandas.DataFrame): DataFrame containing final lead time data.
        calculated_df (pandas.DataFrame): DataFrame containing calculated actual lead time data.

    Returns:
        pandas.DataFrame: DataFrame with lead time differences.
    """

    # Step 1: Find common Material Number and Plant combinations
    material_plant_in_both = set(final_df[['Material Number', 'Plant']].apply(tuple, axis=1)) & set(calculated_df[['Material Number', 'Plant']].apply(tuple, axis=1))
    filtered_final_df = final_df[final_df[['Material Number', 'Plant']].apply(tuple, axis=1).isin(material_plant_in_both)].copy()

    # Step 2: Convert Max and Min Lead Time to days and rename columns
    filtered_final_df['Max Lead Time (Days)'] = filtered_final_df['Max Lead Time'] * 7
    filtered_final_df['Min Lead Time (Days)'] = filtered_final_df['Min Lead Time'] * 7

    # Step 3: Compute the mean of (Max + Min) Lead Time in days
    filtered_final_df['Mean Final Lead Time Days'] = (filtered_final_df['Max Lead Time (Days)'] + filtered_final_df['Min Lead Time (Days)']) / 2

    # Step 4: Compute the mean Actual Lead Time per Material Number from calculated_df
    mean_actual_lead_time = calculated_df.groupby('Material Number')['Actual Lead Time'].mean().reset_index()
    mean_actual_lead_time.rename(columns={'Actual Lead Time': 'Mean Actual Lead Time (Days)'}, inplace=True)

    # Step 5: Merge mean actual lead time back to filtered_final_df
    merged_df = pd.merge(filtered_final_df, mean_actual_lead_time, on='Material Number', how='left')

    # Step 6: Compute Lead Time Difference (Final - Actual)
    merged_df['Lead Time Difference (Days)'] = merged_df['Mean Actual Lead Time (Days)'] - merged_df['Mean Final Lead Time Days'] 

    # Optional cleanup: drop unnecessary columns and re-order if needed
    final_result = merged_df.drop(columns=['Max Lead Time', 'Min Lead Time', 'Mean Final Lead Time Days'])

    return final_result


import plotly.express as px
import pandas as pd

def analyze_and_plot_lead_time_differences_plotly(final_result):
    """
    Analyzes lead time differences and generates Plotly plots, including combined 'Material-Plant' identifier.

    Args:
        final_result (pandas.DataFrame): DataFrame containing lead time difference data.

    Returns:
        tuple: A tuple containing the four generated Plotly figures.
    """

    # Create a combined Material-Plant identifier
    final_result['Material-Plant'] = final_result['Material Number'] + ' - ' + final_result['Plant']

    # Plot 1: Top 10 by absolute difference
    top_10_diff = final_result.reindex(
        final_result['Lead Time Difference (Days)'].abs().sort_values(ascending=False).index
    ).head(10)

    fig1 = px.bar(
        top_10_diff,
        x='Material-Plant',
        y='Lead Time Difference (Days)',
        color='Material Number',
        color_discrete_sequence=px.colors.diverging.Portland,
        title='Top 10 Material-Plant Combinations with the Largest Lead Time Difference',
    )
    fig1.update_layout(xaxis_tickangle=-45)

    # Plot 2: Top 10 over-estimated (late deliveries)
    over_estimated = final_result[final_result['Lead Time Difference (Days)'] > 0].sort_values(
        by='Lead Time Difference (Days)', ascending=False).head(10)

    fig2 = px.bar(
        over_estimated,
        x='Material-Plant',
        y='Lead Time Difference (Days)',
        color='Material Number',
        color_discrete_sequence=px.colors.sequential.Reds,
        title='Top 10 Material-Plant Combinations Delivered Late',
    )
    fig2.update_layout(xaxis_tickangle=-45)

    # Plot 3: Top 10 under-estimated (early deliveries, shown in absolute values)
    under_estimated = final_result[final_result['Lead Time Difference (Days)'] < 0].sort_values(
        by='Lead Time Difference (Days)').head(10).copy()
    under_estimated['Absolute Difference'] = under_estimated['Lead Time Difference (Days)'].abs()

    fig3 = px.bar(
        under_estimated,
        x='Material-Plant',
        y='Absolute Difference',
        color='Material Number',
        color_discrete_sequence=px.colors.sequential.Teal,
        title='Top 10 Material-Plant Combinations Delivered Early',
    )
    fig3.update_layout(xaxis_tickangle=-45)

    # Plot 4: Distribution of lead time differences
    fig4 = px.histogram(
        final_result,
        x='Lead Time Difference (Days)',
        nbins=30,
        title='Distribution of Lead Time Differences',
        color_discrete_sequence=['skyblue'],
        marginal='box',  # Optional: adds a small box plot on top
    )
    fig4.update_layout(bargap=0.2)

    return fig1, fig2, fig3, fig4
