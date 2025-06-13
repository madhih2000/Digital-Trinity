import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go 

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

def calculate_lead_time_summary_v2(df):
    """
    Calculates the maximum and minimum lead times for each material number from an Excel file.

    Args:
        file_path (str): The path to the Excel file.

    Returns:
        pandas.DataFrame: A DataFrame containing the plant, site, material group,
                          material number, supplier, maximum lead time, and minimum lead time.
                          Returns None if an error occurs during file loading.
    """

    lead_time_col = "Lead Time\n(Week)"
    grouped = df.groupby('Material Number')
    result = []

    for material_num, group in grouped:
        plant = group['Plant'].iloc[0]
        site = group['Site'].iloc[0]
        material_group = group['Material Group'].iloc[0]
        supplier = group['Supplier'].iloc[0]
        lead_times = group[lead_time_col].iloc[0]

        if lead_times: #prevent error if lead_times is 0
            max_lead_time = lead_times
            min_lead_time = lead_times
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

def plot_supplier_lead_time_analysis(final_df):
    """
    Analyzes supplier lead time performance and generates interactive plots.

    Calculates the average lead time difference per supplier from the input
    DataFrame and displays three Plotly charts:
    1. Top 5 suppliers (earliest average delivery).
    2. Bottom 5 suppliers (latest average delivery).
    3. Histogram distribution of average lead time differences for all suppliers.

    Args:
        dataframe (pd.DataFrame): A pandas DataFrame containing supplier lead time data.
                                  Must include 'Supplier' and
                                  'Lead Time Difference (Days)' columns.

    Raises:
        ValueError: If the input is not a pandas DataFrame.
        KeyError: If the required columns ('Supplier', 'Lead Time Difference (Days)')
                  are not found in the DataFrame.
    """
    # --- Input Validation ---
    if not isinstance(final_df, pd.DataFrame):
        raise ValueError("Input 'dataframe' must be a pandas DataFrame.")

    required_columns = ['Supplier', 'Lead Time Difference (Days)']
    if not all(col in final_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in final_df.columns]
        raise KeyError(f"Missing required columns in DataFrame: {missing_cols}")

    # --- Analysis ---
    print("Performing supplier lead time analysis...")

    # 1. Calculate average lead time difference per supplier
    # Drop suppliers with no valid lead time difference data before calculating mean
    supplier_performance = final_df.dropna(subset=['Lead Time Difference (Days)']) \
                                   .groupby('Supplier')['Lead Time Difference (Days)'] \
                                   .mean() \
                                   .sort_values()

    if supplier_performance.empty:
        print("No valid supplier performance data found after calculations. Cannot generate plots.")
        return

    # 2. Get Top 5 (Early) and Bottom 5 (Late) Suppliers
    # Handle cases with fewer than 5 suppliers
    n_suppliers = len(supplier_performance)
    top_n = min(5, n_suppliers)
    bottom_n = min(5, n_suppliers)

    top_suppliers = supplier_performance.head(top_n)
    bottom_suppliers = supplier_performance.tail(bottom_n)

    print(f"Found {n_suppliers} suppliers with calculated average lead times.")
    print(f"Identifying top {top_n} and bottom {bottom_n} performers.")

    # --- Plotting with Plotly ---

    # Plot 1: Top Performing Suppliers (Early)
    if not top_suppliers.empty:
        top_df = top_suppliers.reset_index().sort_values(by='Lead Time Difference (Days)', ascending=True)
        fig_top = px.bar(
            top_df,
            x='Lead Time Difference (Days)',
            y='Supplier',
            orientation='h',
            title=f'Top {top_n} Suppliers (Delivering Earliest on Average)',
            labels={'Lead Time Difference (Days)': 'Avg Lead Time Diff (Days) [Negative = Early]', 'Supplier': 'Supplier'},
            text='Lead Time Difference (Days)',
            color='Lead Time Difference (Days)',
            color_continuous_scale=px.colors.sequential.Viridis_r
        )
        fig_top.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_top.update_layout(yaxis={'categoryorder':'total ascending'})
    else:
        print("Skipping Top Suppliers plot (no data).")


    # Plot 2: Bottom Performing Suppliers (Late)
    if not bottom_suppliers.empty:
        bottom_df = bottom_suppliers.reset_index().sort_values(by='Lead Time Difference (Days)', ascending=False)
        fig_bottom = px.bar(
            bottom_df,
            x='Lead Time Difference (Days)',
            y='Supplier',
            orientation='h',
            title=f'Bottom {bottom_n} Suppliers (Delivering Latest on Average)',
            labels={'Lead Time Difference (Days)': 'Avg Lead Time Diff (Days) [Positive = Late]', 'Supplier': 'Supplier'},
            text='Lead Time Difference (Days)',
            color='Lead Time Difference (Days)',
            color_continuous_scale=px.colors.sequential.Magma
        )
        fig_bottom.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_bottom.update_layout(yaxis={'categoryorder':'total descending'})
    else:
        print("Skipping Bottom Suppliers plot (no data).")

    # Plot 3: Distribution of Average Lead Time Difference for All Suppliers
    if not supplier_performance.empty:
        supplier_performance_df = supplier_performance.reset_index()
        fig_dist = px.histogram(
            supplier_performance_df,
            x='Lead Time Difference (Days)',
            marginal="rug",
            title='Distribution of Average Lead Time Difference Across All Suppliers',
            labels={'Lead Time Difference (Days)': 'Average Lead Time Difference (Days)'}
        )
        # Add a vertical line at x=0
        fig_dist.add_vline(
            x=0,
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="On Time (0)",
            annotation_position="top right"
        )
        fig_dist.update_layout(bargap=0.1)
    else:
        print("Skipping Overall Distribution plot (no data).")

    return fig_top, fig_bottom, fig_dist
