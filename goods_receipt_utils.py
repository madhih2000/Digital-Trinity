import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import llm_reasoning

def overall_GR_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall goods receipt patterns, applying a common set of filters to both graphs.
    Top N filter works differently for transaction and goods receipt graphs.
    """

    # Make a copy of the dataframe to avoid modifying the original
    df_filtered = df.copy()

    # -------------------------------------------------------------------
    # GLOBAL FILTERS (Apply to both graphs)
    # -------------------------------------------------------------------

    st.header("Global Filters")

    # Row 1: Plant and Site
    global_filter_row1 = st.columns(2) # Split into two columns

    with global_filter_row1[0]:
        available_plants = sorted(df_filtered['Plant'].unique().tolist())
        selected_plants = st.multiselect("Select Plants", available_plants, default=available_plants, key="plant_filter")
        df_filtered = df_filtered[df_filtered['Plant'].isin(selected_plants)]

    with global_filter_row1[1]:
        available_sites = sorted(df_filtered['Site'].unique().tolist())
        selected_sites = st.multiselect("Select Sites", available_sites, default=available_sites, key="site_filter")
        df_filtered = df_filtered[df_filtered['Site'].isin(selected_sites)]

    # Row 2: Vendor
    df_filtered['Vendor Number'] = df_filtered['Vendor Number'].fillna('Unknown')
    df_filtered['Vendor Number'] = df_filtered['Vendor Number'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Vendor_') else x)
    available_vendors = sorted(df_filtered['Vendor Number'].unique().tolist())
    selected_vendors = st.multiselect("Select Vendors", available_vendors, default=available_vendors)
    df_filtered = df_filtered[df_filtered['Vendor Number'].isin(selected_vendors)]

    # Row 3: Date Range
    global_filter_row3 = st.columns(1) # Date range alone in this row.
    with global_filter_row3[0]:
        try:
            df_filtered['Pstng Date'] = pd.to_datetime(df_filtered['Pstng Date'], format='%d/%m/%Y %I:%M:%S %p', errors='raise')
        except ValueError as e:
            st.error(f"Error converting 'Pstng Date' column to datetime. Ensure the date format is consistent. Error: {e}")
            return
        except KeyError:
            st.error("The column 'Pstng Date' was not found in the DataFrame.")
            return

        min_date = df_filtered['Pstng Date'].min()
        max_date = df_filtered['Pstng Date'].max()
        date_range = st.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[(df_filtered['Pstng Date'] >= pd.to_datetime(start_date)) & (df_filtered['Pstng Date'] <= pd.to_datetime(end_date))]

    # Row 4: Top N Material Selection
    top_n = st.selectbox("Select Top N Materials", [5, 10, 15, 'All'], index=1)



    # -------------------------------------------------------------------
    # GRAPH 1 - Number of Transactions
    # -------------------------------------------------------------------

    # Use the globally filtered DataFrame
    df_transactions = df_filtered.copy()

    # Top N Filtering for Transactions
    material_counts = df_transactions[material_column].value_counts().reset_index()
    material_counts.columns = [material_column, 'Transaction Count']
    material_counts = material_counts.sort_values(by='Transaction Count', ascending=False)

    if top_n != 'All':
        top_n_int = int(top_n)
        top_materials_trans = material_counts[material_column].head(top_n_int).tolist()
        df_transactions = df_transactions[df_transactions[material_column].isin(top_materials_trans)]


    # Visualization - Number of Transactions
    transaction_counts = df_transactions[material_column].value_counts().reset_index()
    transaction_counts.columns = [material_column, 'Transaction Count']
    fig_transactions = px.bar(transaction_counts, x=material_column, y='Transaction Count',
                              title=f'Number of Transactions per {material_column}')
    st.plotly_chart(fig_transactions)


    # -------------------------------------------------------------------
    # GRAPH 2 - Overall Goods Receipt
    # -------------------------------------------------------------------


    # Top N Filtering for Goods Receipt
    #Assumes there is a column named 'Quantity'
    material_consumption_sum = df_transactions.groupby(material_column)['Quantity'].sum().abs().reset_index() #Absolute sum
    material_consumption_sum = material_consumption_sum.sort_values(by='Quantity', ascending=False)

    if top_n != 'All':
        top_n_int = int(top_n) #To convert from string to int
        top_materials_cons = material_consumption_sum[material_column].head(top_n_int).tolist()
        df_transactions = df_transactions[df_transactions[material_column].isin(top_materials_cons)]

    # Visualization - Overall Consumption
    fig_overall = px.bar(material_consumption_sum, x=material_column, y='Quantity',
                         title=f'Overall Goods Receipt by {material_column}')
    st.plotly_chart(fig_overall)


    # -------------------------------------------------------------------
    # Data Display (For Debugging Purposes)
    # -------------------------------------------------------------------

    #st.write("Final Dataframe after Global Filters:")
    #st.dataframe(df_filtered)

    return df_filtered, top_n #Important, must return for the following code to work

def outlier_detection(df, top_n, material_column='Material Number'):
    """
    Detects and visualizes outliers, including percentiles and highlighting high/low usage.

    Args:
        df: The input DataFrame.
        top_n: Number of top materials by variance to display in the plot.
        material_column: The column containing material identifiers (default: 'Material Number').
    """

    # Calculate outliers using IQR
    outliers = df.groupby(material_column)['Quantity'].apply(detect_outliers_iqr).reset_index(name='Outlier Quantity')
    outliers = outliers[[material_column, 'Outlier Quantity']]  # Remove 'level_1'

    # Calculate percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
    percentiles = df.groupby(material_column)['Quantity'].describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()

    # Merge percentiles with outliers
    outliers_with_percentiles = pd.merge(outliers, percentiles, on=material_column)

    # Add a "Type" column to indicate high or low usage 
    outliers_with_percentiles['Type'] = outliers_with_percentiles.apply(
        lambda row: 'High' if row['Outlier Quantity'] > row['75%'] else ('Low' if row['Outlier Quantity'] < row['25%'] else 'Normal'), axis=1
    )

    # Compute variance per material
    variance_df = df.groupby(material_column)['Quantity'].var().reset_index().rename(columns={'Quantity': 'Variance'})
    
    # Sort by variance in descending order
    variance_df_sorted = variance_df.sort_values(by='Variance', ascending=False)

    # Select only top_n materials
    top_materials = variance_df_sorted.head(top_n)[material_column]

    # Filter original DataFrame to include only top_n materials
    df_filtered = df[df[material_column].isin(top_materials)]

    # Display sorted variance DataFrame
    #st.write("Top Materials by Variance:", variance_df_sorted.head(top_n))

    # Convert column to categorical with the new sorted order
    df_filtered[material_column] = pd.Categorical(df_filtered[material_column], categories=top_materials, ordered=True)

    # Box Plot for Outlier Visualization (Filtered & Sorted)
    fig_box = px.box(df_filtered, x=material_column, y='Quantity', title=f'Materials by Variance')
    
    st.plotly_chart(fig_box)

    llm_reasoning.explain_box_plot_with_groq_goods_receipt(df_filtered)

def plot_quantity_distribution(data):
    """
    Plot the distribution of the 'Quantity' column.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Quantity' column.
    
    Returns:
    - None: Displays the plot in the Streamlit app.
    """
    fig = px.histogram(data, x='Quantity', title="Distribution of Quantities")
    st.plotly_chart(fig)


def detect_outliers_iqr(data):
    """
    Detects outliers in a pandas Series using the IQR method.

    Args:
        data: The input pandas Series.

    Returns:
        A pandas Series containing the outliers.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers



def specific_material_analysis(df, material_column='Material Number'):
    """
    Analyzes goods receipt patterns for a specific material, including goods receipt trend, seasonal subseries,
    and filters by Site, Plant, and Vendor.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    st.markdown(
    """
    <hr style="
        border: none;
        height: 4px;
        background: linear-gradient(to right, #00FF00, #0000FF);
        margin: 20px 0;">
    """,
    unsafe_allow_html=True
    )

    st.subheader("Material-Level Analysis")  # Add a section title

    selected_material = st.selectbox(f"Select a {material_column}", df[material_column].unique())
    filtered_material_data = df[df[material_column] == selected_material].copy()  # Create a copy

    # -------------------------------------------------------------------
    # Filters for Site, Plant, and Vendor
    # -------------------------------------------------------------------

    st.subheader("Filters")  # Add a section title for filters

    filter_row = st.columns(3)  # Create a horizontal layout for filters

    with filter_row[0]:
        available_plants = sorted(filtered_material_data['Plant'].unique().tolist())
        selected_plants = st.multiselect("Select Plants", available_plants, default=available_plants, key="plant_key_specific")
        filtered_material_data = filtered_material_data[filtered_material_data['Plant'].isin(selected_plants)]

    with filter_row[1]:
        available_sites = sorted(filtered_material_data['Site'].unique().tolist())
        selected_sites = st.multiselect("Select Sites", available_sites, default=available_sites, key="site_key_specific")
        filtered_material_data = filtered_material_data[filtered_material_data['Site'].isin(selected_sites)]

    with filter_row[2]:
        # Handle missing/invalid Vendor Numbers
        filtered_material_data['Vendor Number'] = filtered_material_data['Vendor Number'].fillna('Unknown')
        filtered_material_data['Vendor Number'] = filtered_material_data['Vendor Number'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Vendor_') else x)

        available_vendors = sorted(filtered_material_data['Vendor Number'].unique().tolist())
        selected_vendors = st.multiselect("Select Vendors", available_vendors, default=available_vendors)
        filtered_material_data = filtered_material_data[filtered_material_data['Vendor Number'].isin(selected_vendors)]

    # -------------------------------------------------------------------
    # Time Series of Goods Receipt (Trend)
    # -------------------------------------------------------------------

    try:
        filtered_material_data['Pstng Date'] = pd.to_datetime(filtered_material_data['Pstng Date'], format='%d/%m/%Y %I:%M:%S %p', errors='raise')
    except (ValueError, KeyError) as e:
        st.error(f"Error converting 'Pstng Date' to datetime: {e}. Ensure the column exists and contains valid date values.")
        return

    min_date = filtered_material_data['Pstng Date'].min().date()
    max_date = filtered_material_data['Pstng Date'].max().date()

    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame({'Pstng Date': date_range})

    filtered_time_data = filtered_material_data[
        (filtered_material_data['Pstng Date'].dt.date >= start_date) &
        (filtered_material_data['Pstng Date'].dt.date <= end_date)
    ]

    aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    if aggregation_level == 'Daily':
        aggregated_data = filtered_time_data.groupby('Pstng Date')['Quantity'].sum().reset_index()
        aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
        aggregated_data = pd.merge(date_df, aggregated_data, on='Pstng Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(filtered_time_data['Pstng Date'].dt.date).size().reset_index(name='Transaction Count')
        transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
        transaction_counts = pd.merge(date_df, transaction_counts, on='Pstng Date', how='left').fillna(0)

    elif aggregation_level == 'Weekly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Quantity'].sum().reset_index()
        date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
        date_df_weekly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='W')).min().reset_index()
        aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
        aggregated_data = pd.merge(date_df_weekly, aggregated_data, on='Pstng Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W')).size().reset_index(name='Transaction Count')
        transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
        transaction_counts = pd.merge(date_df_weekly, transaction_counts, on='Pstng Date', how='left').fillna(0)

    elif aggregation_level == 'Monthly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M'))['Quantity'].sum().reset_index()
        date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
        date_df_monthly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='M')).min().reset_index()
        aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
        aggregated_data = pd.merge(date_df_monthly, aggregated_data, on='Pstng Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M')).size().reset_index(name='Transaction Count')
        transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
        transaction_counts = pd.merge(date_df_monthly, transaction_counts, on='Pstng Date', how='left').fillna(0)

    elif aggregation_level == 'Quarterly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q'))['Quantity'].sum().reset_index()
        date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
        date_df_quarterly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='Q')).min().reset_index()
        aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
        aggregated_data = pd.merge(date_df_quarterly, aggregated_data, on='Pstng Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q')).size().reset_index(name='Transaction Count')
        transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
        transaction_counts = pd.merge(date_df_quarterly, transaction_counts, on='Pstng Date', how='left').fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=aggregated_data['Pstng Date'], y=aggregated_data['Quantity'], name='Quantity', yaxis='y1'))
    fig.add_trace(go.Scatter(x=transaction_counts['Pstng Date'], y=transaction_counts['Transaction Count'], name='Transaction Count', yaxis='y2'))

    fig.update_layout(
        title=f'Goods Receipt Trend and Transaction Count ({aggregation_level}) for {selected_material}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        yaxis2=dict(
            title='Transaction Count',
            overlaying='y',
            side='right'
        )
    )

    st.plotly_chart(fig)

    # -------------------------------------------------------------------
    # Quantity by Plant over Time (Bar Chart)
    # -------------------------------------------------------------------

    st.subheader("Goods Receipt Quantity by Plant Over Time")

    if aggregation_level == 'Daily':
        plant_aggregated_data = filtered_time_data.groupby(['Pstng Date', 'Plant'])['Quantity'].sum().reset_index()
        plant_aggregated_data['Pstng Date'] = pd.to_datetime(plant_aggregated_data['Pstng Date'])
        date_plant_df = pd.MultiIndex.from_product([date_df['Pstng Date'].tolist(), available_plants], names=['Pstng Date', 'Plant']).to_frame(index=False)
        plant_aggregated_data = pd.merge(date_plant_df, plant_aggregated_data, on=['Pstng Date', 'Plant'], how='left').fillna(0)

    elif aggregation_level == 'Weekly':
        plant_aggregated_data = filtered_time_data.groupby([pd.Grouper(key='Pstng Date', freq='W'), 'Plant'])['Quantity'].sum().reset_index()
        date_plant_df = pd.MultiIndex.from_product([date_df_weekly['Pstng Date'].tolist(), available_plants], names=['Pstng Date', 'Plant']).to_frame(index=False)
        plant_aggregated_data['Pstng Date'] = pd.to_datetime(plant_aggregated_data['Pstng Date'])
        plant_aggregated_data = pd.merge(date_plant_df, plant_aggregated_data, on=['Pstng Date', 'Plant'], how='left').fillna(0)

    elif aggregation_level == 'Monthly':
        plant_aggregated_data = filtered_time_data.groupby([pd.Grouper(key='Pstng Date', freq='M'), 'Plant'])['Quantity'].sum().reset_index()
        date_plant_df = pd.MultiIndex.from_product([date_df_monthly['Pstng Date'].tolist(), available_plants], names=['Pstng Date', 'Plant']).to_frame(index=False)
        plant_aggregated_data['Pstng Date'] = pd.to_datetime(plant_aggregated_data['Pstng Date'])
        plant_aggregated_data = pd.merge(date_plant_df, plant_aggregated_data, on=['Pstng Date', 'Plant'], how='left').fillna(0)

    elif aggregation_level == 'Quarterly':
        plant_aggregated_data = filtered_time_data.groupby([pd.Grouper(key='Pstng Date', freq='Q'), 'Plant'])['Quantity'].sum().reset_index()
        date_plant_df = pd.MultiIndex.from_product([date_df_quarterly['Pstng Date'].tolist(), available_plants], names=['Pstng Date', 'Plant']).to_frame(index=False)
        plant_aggregated_data['Pstng Date'] = pd.to_datetime(plant_aggregated_data['Pstng Date'])
        plant_aggregated_data = pd.merge(date_plant_df, plant_aggregated_data, on=['Pstng Date', 'Plant'], how='left').fillna(0)

    fig_plant = go.Figure()

    for plant in available_plants:
        plant_data = plant_aggregated_data[plant_aggregated_data['Plant'] == plant]
        fig_plant.add_trace(go.Bar(x=plant_data['Pstng Date'], y=plant_data['Quantity'], name=plant))

    fig_plant.update_layout(
        title=f'Goods Receipt Quantity by Plant ({aggregation_level}) for {selected_material}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        barmode='group'  # Group bars for each plant
    )

    st.plotly_chart(fig_plant)

def plot_quantity_boxplot(data):
    """
    Plot a boxplot to detect outliers in the 'Quantity' column.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Quantity' column.
    
    Returns:
    - None: Displays the boxplot in the Streamlit app.
    """
    fig = px.box(data, y='Quantity', title="Boxplot of Quantity (Outlier Detection)")
    st.plotly_chart(fig)


def plot_vendor_distribution(data):
    """
    Plot the distribution of 'Vendor Number' (number of orders per vendor).
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Vendor Number' column.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    vendor_counts = data['Vendor Number'].value_counts()
    fig = px.bar(vendor_counts, x=vendor_counts.index, y=vendor_counts.values, 
                 title="Distribution of Vendors", labels={'x': 'Vendor Number', 'y': 'Count'})
    st.plotly_chart(fig)

def plot_quantity_trends_per_vendor(data):
    """
    Plot the trends of 'Quantity' over time per 'Vendor Number' based on 'Pstng Date'.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M'), 'Vendor Number'])['Quantity'].sum().reset_index()

    # Convert Period back to string or datetime before plotting
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)  # Converts '2024-01' to '2024-01'

    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', color='Vendor Number', 
                  title="Quantity Trends Over Time by Vendor",
                  labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})

    st.plotly_chart(fig)



def plot_purchase_frequency(data):
    """
    Plots the frequency of purchases over time.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby(data['Pstng Date'].dt.to_period('M')).size().reset_index(name='Frequency')

    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)

    fig = px.bar(data_grouped, x='Pstng Date', y='Frequency', 
                 title="Purchase Frequency Over Time", 
                 labels={'Pstng Date': 'Posting Date', 'Frequency': 'Number of Purchases'})

    st.plotly_chart(fig)


def plot_quantity_vs_pstng_date(data):
    """
    Plot a scatter plot of 'Quantity' against 'Posting Date' with color-coding by 'Vendor Number'.
    
    Args:
    - data (pandas.DataFrame): The dataset containing 'Quantity', 'Pstng Date', and 'Vendor Number' columns.
    
    Returns:
    - None: Displays the scatter plot in the Streamlit app.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    fig = px.scatter(data, x='Pstng Date', y='Quantity', color='Vendor Number', 
                     title="Quantity vs. Posting Date", labels={'Pstng Date': 'Posting Date', 'Quantity': 'Quantity'})
    st.plotly_chart(fig)

def plot_quantity_per_plant(data):
    """
    Plot the total quantity per plant.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Plant' and 'Quantity' columns.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    plant_counts = data.groupby('Plant')['Quantity'].sum().reset_index()
    fig = px.bar(plant_counts, x='Plant', y='Quantity', title="Quantity Per Plant", labels={'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_quantity_per_batch(data):
    """
    Plot the total quantity per batch.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Batch' and 'Quantity' columns.
    
    Returns:
    - None: Displays the bar chart in the Streamlit app.
    """
    batch_counts = data.groupby('Batch')['Quantity'].sum().reset_index()
    fig = px.bar(batch_counts, x='Batch', y='Quantity', title="Quantity Per Batch", labels={'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_sled_bbd_distribution(data):
    """
    Plot the distribution of 'SLED/BBD' dates.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'SLED/BBD' column.
    
    Returns:
    - None: Displays the histogram in the Streamlit app.
    """
    data['SLED/BBD'] = pd.to_datetime(data['SLED/BBD'])
    fig = px.histogram(data, x='SLED/BBD', title="SLED/BBD Distribution", labels={'SLED/BBD': 'SLED/BBD Date'})
    st.plotly_chart(fig)

def plot_missing_values(data):
    """
    Plot the count of missing values per column.
    
    Args:
    - data (pandas.DataFrame): The dataset to analyze for missing values.
    
    Returns:
    - None: Displays the missing values distribution bar chart in the Streamlit app.
    """
    missing_data = data.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    fig = px.bar(missing_data, x=missing_data.index, y=missing_data.values, 
                 title="Missing Data Distribution", labels={'x': 'Columns', 'y': 'Missing Values'})
    st.plotly_chart(fig)

def plot_time_based_trends(data):
    """
    Plot the trends of 'Quantity' over time based on 'Pstng Date'.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M').dt.to_timestamp()])['Quantity'].sum().reset_index()

    # Convert Period to string (or datetime) before plotting
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)  # Converts '2024-01' to '2024-01'

    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', 
                  title="Quantity Trends Over Time", 
                  labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})

    st.plotly_chart(fig)

def plot_quantity_trends_per_plant(data):
    """
    Plot the trends of 'Quantity' over time per 'Plant' based on 'Pstng Date'.
    
    Args:
    - data (pandas.DataFrame): The dataset containing the 'Pstng Date', 'Quantity', and 'Plant' columns.
    
    Returns:
    - None: Displays the line chart in the Streamlit app.
    """
    data['Pstng Date'] = pd.to_datetime(data['Pstng Date'])
    data_grouped = data.groupby([data['Pstng Date'].dt.to_period('M'), 'Plant'])['Quantity'].sum().reset_index()
    data_grouped['Pstng Date'] = data_grouped['Pstng Date'].astype(str)
    fig = px.line(data_grouped, x='Pstng Date', y='Quantity', color='Plant', 
                  title="Quantity Trends Over Time by Plant", labels={'Pstng Date': 'Posting Date', 'Quantity': 'Total Quantity'})
    st.plotly_chart(fig)

def plot_sled_bbd_vs_quantity(data):
    """
    Plot 'SLED/BBD' vs 'Quantity' to analyze if there is any correlation between the shelf life or batch dates and quantities.
    
    Args:
    - data (pandas.DataFrame): The dataset containing 'SLED/BBD' and 'Quantity' columns.
    
    Returns:
    - None: Displays the scatter plot in the Streamlit app.
    """
    data['SLED/BBD'] = pd.to_datetime(data['SLED/BBD'])
    fig = px.scatter(data, x='SLED/BBD', y='Quantity', 
                     title="SLED/BBD vs Quantity", 
                     labels={'SLED/BBD': 'SLED/BBD Date', 'Quantity': 'Quantity'})
    st.plotly_chart(fig)


