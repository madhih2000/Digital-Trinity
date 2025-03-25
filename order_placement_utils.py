import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import llm_reasoning

# Function to preprocess the uploaded file
def preprocess_order_data(file):
    if file:
        data = pd.read_excel(file)
        data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
        data.columns = data.columns.str.strip()
        # Ensure correct data types
        data['Order Quantity'] = pd.to_numeric(data['Order Quantity'], errors='coerce')
        return data
    return None


def overall_orderplacement_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall orderplacement patterns, applying a common set of filters to both graphs.
    Top N filter works differently for transaction and order placement graphs.
    """

    # Make a copy of the dataframe to avoid modifying the original
    df_filtered = df.copy()

    # -------------------------------------------------------------------
    # GLOBAL FILTERS (Apply to both graphs)
    # -------------------------------------------------------------------

    st.header("Global Filters")

    # Row 1: Plant

    available_plants = sorted(df_filtered['Plant'].unique().tolist())
    selected_plants = st.multiselect("Select Plants", available_plants, default=available_plants, key="plant_filter")
    df_filtered = df_filtered[df_filtered['Plant'].isin(selected_plants)]
    
    # Row 2: Supplier
    df_filtered['Supplier'] = df_filtered['Supplier'].fillna('Unknown')
    df_filtered['Supplier'] = df_filtered['Supplier'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Supplier_') else x)
    available_suppliers = sorted(df_filtered['Supplier'].unique().tolist())
    selected_suppliers = st.multiselect("Select Suppliers", available_suppliers, default=available_suppliers)
    df_filtered = df_filtered[df_filtered['Supplier'].isin(selected_suppliers)]

    # Row 3: Vendor
    df_filtered['Vendor Number'] = df_filtered['Vendor Number'].fillna('Unknown')
    df_filtered['Vendor Number'] = df_filtered['Vendor Number'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Vendor_') else x)
    available_vendors = sorted(df_filtered['Vendor Number'].unique().tolist())
    selected_vendors = st.multiselect("Select Vendors", available_vendors, default=available_vendors)
    df_filtered = df_filtered[df_filtered['Vendor Number'].isin(selected_vendors)]

    # Row 4: Date Range
    global_filter_row3 = st.columns(1) # Date range alone in this row.
    with global_filter_row3[0]:
        try:
            df_filtered['Document Date'] = pd.to_datetime(df_filtered['Document Date'], format='%d/%m/%Y %I:%M:%S %p', errors='raise')
        except ValueError as e:
            st.error(f"Error converting 'Document Date' column to datetime. Ensure the date format is consistent. Error: {e}")
            return
        except KeyError:
            st.error("The column 'Document Date' was not found in the DataFrame.")
            return

        min_date = df_filtered['Document Date'].min()
        max_date = df_filtered['Document Date'].max()
        date_range = st.date_input("Select Date Range", value=[min_date, max_date], min_value=min_date, max_value=max_date)

        if len(date_range) == 2:
            start_date, end_date = date_range
            df_filtered = df_filtered[(df_filtered['Document Date'] >= pd.to_datetime(start_date)) & (df_filtered['Document Date'] <= pd.to_datetime(end_date))]

    # Row 5: Top N Material Selection
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
    # GRAPH 2 - Overall Order Placement
    # -------------------------------------------------------------------


    # Top N Filtering for Order Placement
    #Assumes there is a column named 'Order Quantity'
    material_consumption_sum = df_transactions.groupby(material_column)['Order Quantity'].sum().abs().reset_index() #Absolute sum
    material_consumption_sum = material_consumption_sum.sort_values(by='Order Quantity', ascending=False)

    if top_n != 'All':
        top_n_int = int(top_n) #To convert from string to int
        top_materials_cons = material_consumption_sum[material_column].head(top_n_int).tolist()
        df_transactions = df_transactions[df_transactions[material_column].isin(top_materials_cons)]

    # Visualization - Overall Consumption
    fig_overall = px.bar(material_consumption_sum, x=material_column, y='Order Quantity',
                         title=f'Overall Order Placement by {material_column}')
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
    outliers = df.groupby(material_column)['Order Quantity'].apply(detect_outliers_iqr).reset_index(name='Outlier Quantity')
    outliers = outliers[[material_column, 'Outlier Quantity']]  # Remove 'level_1'

    # Calculate percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
    percentiles = df.groupby(material_column)['Order Quantity'].describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()

    # Merge percentiles with outliers
    outliers_with_percentiles = pd.merge(outliers, percentiles, on=material_column)

    # Add a "Type" column to indicate high or low usage 
    outliers_with_percentiles['Type'] = outliers_with_percentiles.apply(
        lambda row: 'High' if row['Outlier Quantity'] > row['75%'] else ('Low' if row['Outlier Quantity'] < row['25%'] else 'Normal'), axis=1
    )

    # Compute variance per material
    variance_df = df.groupby(material_column)['Order Quantity'].var().reset_index().rename(columns={'Order Quantity': 'Variance'})
    
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
    fig_box = px.box(df_filtered, x=material_column, y='Order Quantity', title=f'Materials by Variance')
    st.plotly_chart(fig_box)

    llm_reasoning.explain_box_plot_with_groq_orderplacement(df_filtered)

def overall_order_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall order patterns.

    Args:
        df: The input DataFrame.
        material_column: The column containing material identifiers (default: 'Material Number').
    """
    # Frequency of Orders
    order_counts = df[material_column].value_counts().reset_index()
    order_counts.columns = [material_column, 'Order Count']
    fig_orders = px.bar(order_counts, x=material_column, y='Order Count',
                        title=f'Number of Orders per {material_column}')
    st.plotly_chart(fig_orders)

    # Volume of Orders
    material_orders = df.groupby(material_column)['Order Quantity'].sum().reset_index()
    fig_overall = px.bar(material_orders, x=material_column, y='Order Quantity',
                        title=f'Overall Order Quantity by {material_column}')
    st.plotly_chart(fig_overall)

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

# def outlier_detection(df, material_column='Material Number'):
#     """
#     Detects and visualizes outliers, including percentiles and highlighting high/low order quantities.

#     Args:
#         df: The input DataFrame.
#         material_column: The column containing material identifiers (default: 'Material Number').
#     """

#     # Calculate outliers using IQR
#     outliers = df.groupby(material_column)['Order Quantity'].apply(detect_outliers_iqr).reset_index(name='Outlier Quantity')
#     outliers = outliers[[material_column, 'Outlier Quantity']]  # Remove 'level_1'

#     # Calculate percentiles (e.g., 10th, 25th, 50th, 75th, 90th)
#     percentiles = df.groupby(material_column)['Order Quantity'].describe(percentiles=[.1, .25, .5, .75, .9]).reset_index()

#     # Merge percentiles with outliers
#     outliers_with_percentiles = pd.merge(outliers, percentiles, on=material_column)

#     # Add a "Type" column to indicate high or low usage
#     outliers_with_percentiles['Type'] = outliers_with_percentiles.apply(
#         lambda row: 'High' if row['Outlier Quantity'] > row['75%'] else ('Low' if row['Outlier Quantity'] < row['25%'] else 'Normal'), axis=1
#     )

#     # Color coding for "High" and "Low" in the table
#     def color_coding(val):
#         if val == 'High':
#             color = 'red'  # Color "High" values red
#         elif val == 'Low':
#             color = 'blue'  # Color "Low" values blue
#         else:
#             color = 'black'  # Default color
#         return f'color: {color}'

#     # Format the table to display integers for relevant columns
#     styled_table = outliers_with_percentiles.style.applymap(color_coding, subset=['Type']).format({
#         'Outlier Quantity': "{:.0f}".format,
#         'count': "{:.0f}".format,
#         'min': "{:.0f}".format,
#         '10%': "{:.0f}".format,
#         '25%': "{:.0f}".format,
#         '50%': "{:.0f}".format,
#         '75%': "{:.0f}".format,
#         '90%': "{:.0f}".format,
#         'max': "{:.0f}".format
#     })

#     st.write(f"Outliers (IQR Method) with Percentiles for {material_column}:")
#     st.write(styled_table)

#     # Box Plots for Outlier Visualization
#     fig_box = px.box(df, x=material_column, y='Order Quantity', title=f'Box Plot of Order Quantity by {material_column}')
#     st.plotly_chart(fig_box)


def supplier_order_analysis(df):
    """
    Analyzes order patterns for each supplier.

    Args:
        df: The input DataFrame.
    """
    # Order Quantity by Supplier
    supplier_orders = df.groupby('Supplier')['Order Quantity'].sum().reset_index()
    fig_supplier = px.bar(supplier_orders, x='Supplier', y='Order Quantity', title='Order Quantity by Supplier')
    st.plotly_chart(fig_supplier)

def specific_material_analysis(df, material_column='Material Number'):
    """
    Analyzes order placement patterns for a specific material, including order placement trend, seasonal subseries,
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

    available_plants = sorted(filtered_material_data['Plant'].unique().tolist())
    selected_plants = st.multiselect("Select Plants", available_plants, default=available_plants, key="plant_key_specific")
    filtered_material_data = filtered_material_data[filtered_material_data['Plant'].isin(selected_plants)]

    filtered_material_data['Supplier'] = filtered_material_data['Supplier'].fillna('Unknown')
    filtered_material_data['Supplier'] = filtered_material_data['Supplier'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Supplier_') else x)
    available_suppliers = sorted(filtered_material_data['Supplier'].unique().tolist())
    selected_suppliers = st.multiselect("Select Suppliers", available_suppliers, default=available_suppliers, key="supplier_key_specific")
    filtered_material_data = filtered_material_data[filtered_material_data['Supplier'].isin(selected_suppliers)]

    # Handle missing/invalid Vendor Numbers
    filtered_material_data['Vendor Number'] = filtered_material_data['Vendor Number'].fillna('Unknown')
    filtered_material_data['Vendor Number'] = filtered_material_data['Vendor Number'].apply(lambda x: 'Unknown' if not isinstance(x, str) or not x.startswith('Vendor_') else x)
    available_vendors = sorted(filtered_material_data['Vendor Number'].unique().tolist())
    selected_vendors = st.multiselect("Select Vendors", available_vendors, default=available_vendors, key="vendor_key_specific")
    filtered_material_data = filtered_material_data[filtered_material_data['Vendor Number'].isin(selected_vendors)]

    # -------------------------------------------------------------------
    # Time Series of Order Placement (Trend)
    # -------------------------------------------------------------------

    try:
        filtered_material_data['Document Date'] = pd.to_datetime(filtered_material_data['Document Date'], format='%d/%m/%Y %I:%M:%S %p', errors='raise')
    except (ValueError, KeyError) as e:
        st.error(f"Error converting 'Pstng Date' to datetime: {e}. Ensure the column exists and contains valid date values.")
        return

    min_date = filtered_material_data['Document Date'].min().date()
    max_date = filtered_material_data['Document Date'].max().date()

    start_date, end_date = st.date_input("Select Date Range", [min_date, max_date])

    date_range = pd.date_range(start=start_date, end=end_date)
    date_df = pd.DataFrame({'Document Date': date_range})

    filtered_time_data = filtered_material_data[
        (filtered_material_data['Document Date'].dt.date >= start_date) &
        (filtered_material_data['Document Date'].dt.date <= end_date)
    ]

    aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    if aggregation_level == 'Daily':
        aggregated_data = filtered_time_data.groupby('Document Date')['Order Quantity'].sum().reset_index()
        aggregated_data['Document Date'] = pd.to_datetime(aggregated_data['Document Date'])
        aggregated_data = pd.merge(date_df, aggregated_data, on='Document Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(filtered_time_data['Document Date'].dt.date).size().reset_index(name='Transaction Count')
        transaction_counts['Document Date'] = pd.to_datetime(transaction_counts['Document Date'])
        transaction_counts = pd.merge(date_df, transaction_counts, on='Document Date', how='left').fillna(0)

    elif aggregation_level == 'Weekly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='W'))['Order Quantity'].sum().reset_index()
        date_df['Document Date'] = pd.to_datetime(date_df['Document Date'])
        date_df_weekly = date_df.groupby(pd.Grouper(key='Document Date', freq='W')).min().reset_index()
        aggregated_data['Document Date'] = pd.to_datetime(aggregated_data['Document Date'])
        aggregated_data = pd.merge(date_df_weekly, aggregated_data, on='Document Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='W')).size().reset_index(name='Transaction Count')
        transaction_counts['Document Date'] = pd.to_datetime(transaction_counts['Document Date'])
        transaction_counts = pd.merge(date_df_weekly, transaction_counts, on='Document Date', how='left').fillna(0)

    elif aggregation_level == 'Monthly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='M'))['Order Quantity'].sum().reset_index()
        date_df['Document Date'] = pd.to_datetime(date_df['Document Date'])
        date_df_monthly = date_df.groupby(pd.Grouper(key='Document Date', freq='M')).min().reset_index()
        aggregated_data['Document Date'] = pd.to_datetime(aggregated_data['Document Date'])
        aggregated_data = pd.merge(date_df_monthly, aggregated_data, on='Document Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='M')).size().reset_index(name='Transaction Count')
        transaction_counts['Document Date'] = pd.to_datetime(transaction_counts['Document Date'])
        transaction_counts = pd.merge(date_df_monthly, transaction_counts, on='Document Date', how='left').fillna(0)

    elif aggregation_level == 'Quarterly':
        aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='Q'))['Order Quantity'].sum().reset_index()
        date_df['Document Date'] = pd.to_datetime(date_df['Document Date'])
        date_df_quarterly = date_df.groupby(pd.Grouper(key='Document Date', freq='Q')).min().reset_index()
        aggregated_data['Document Date'] = pd.to_datetime(aggregated_data['Document Date'])
        aggregated_data = pd.merge(date_df_quarterly, aggregated_data, on='Document Date', how='left').fillna(0)

        transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Document Date', freq='Q')).size().reset_index(name='Transaction Count')
        transaction_counts['Document Date'] = pd.to_datetime(transaction_counts['Document Date'])
        transaction_counts = pd.merge(date_df_quarterly, transaction_counts, on='Document Date', how='left').fillna(0)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=aggregated_data['Document Date'], y=aggregated_data['Order Quantity'], name='Order Quantity', yaxis='y1'))
    fig.add_trace(go.Scatter(x=transaction_counts['Document Date'], y=transaction_counts['Transaction Count'], name='Transaction Count', yaxis='y2'))

    fig.update_layout(
        title=f'Order Placement Trend and Transaction Count ({aggregation_level}) for {selected_material}',
        xaxis_title='Date',
        yaxis_title='Order Quantity',
        yaxis2=dict(
            title='Transaction Count',
            overlaying='y',
            side='right'
        )
    )

    st.plotly_chart(fig)

# def specific_material_analysis(df, material_column='Material Number'):
#     """
#     Analyzes order patterns for a specific material, including order quantity by plant, vendor, trend, and seasonal subseries.

#     Args:
#         df: The input DataFrame.
#         material_column: The column containing material identifiers (default: 'Material Number').
#     """
#     selected_material = st.selectbox(f"Select a {material_column}", df[material_column].unique())
#     filtered_material_data = df[df[material_column] == selected_material].copy()  # Create a copy

#     # Order Quantity by Plant for Selected Material
#     plant_orders = filtered_material_data.groupby('Plant')['Order Quantity'].sum().reset_index()
#     fig_plant = px.bar(plant_orders, x='Plant', y='Order Quantity', title=f'Order Quantity by Plant for {selected_material}')
#     st.plotly_chart(fig_plant)

#     # Order Quantity by Vendor for Selected Material
#     vendor_orders = filtered_material_data.groupby('Vendor Number')['Order Quantity'].sum().reset_index()
#     fig_vendor = px.bar(vendor_orders, x='Vendor Number', y='Order Quantity', title=f'Order Quantity by Vendor for {selected_material}')
#     st.plotly_chart(fig_vendor)

#     # Time Series of Orders (Trend)
#     if 'Pstng Date' in filtered_material_data.columns:
#         start_date, end_date = st.date_input("Select Date Range",
#                                             [filtered_material_data['Pstng Date'].min().date(),
#                                              filtered_material_data['Pstng Date'].max().date()])

#         filtered_time_data = filtered_material_data[(filtered_material_data['Pstng Date'].dt.date >= start_date) &
#                                                     (filtered_material_data['Pstng Date'].dt.date <= end_date)]

#         aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

#         if aggregation_level == 'Daily':
#             aggregated_data = filtered_time_data.groupby('Pstng Date')['Order Quantity'].sum().reset_index()
#         elif aggregation_level == 'Weekly':
#             aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Order Quantity'].sum().reset_index()
#         elif aggregation_level == 'Monthly':
#             aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M'))['Order Quantity'].sum().reset_index()
#         elif aggregation_level == 'Quarterly':
#             aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q'))['Order Quantity'].sum().reset_index()

#         fig_time_series = px.line(aggregated_data, x='Pstng Date', y='Order Quantity',
#                                     title=f'Order Trend ({aggregation_level}) for {selected_material}')
#         st.plotly_chart(fig_time_series)

#         # Seasonal Subseries Plot (Monthly)
#         if aggregation_level in ['Monthly', 'Quarterly']:
#             filtered_time_data['Month'] = filtered_time_data['Pstng Date'].dt.month
#             fig_seasonal_subseries = px.box(filtered_time_data, x='Month', y='Order Quantity',
#                                                 title=f'Seasonal Subseries Plot (Monthly) for {selected_material}')
#             st.plotly_chart(fig_seasonal_subseries)
#     else:
#         st.write("No 'Pstng Date' column found for time-based analysis.")

def vendor_order_analysis(df):
    """
    Analyzes order patterns for each vendor.

    Args:
        df: The input DataFrame.
    """
    # Order Quantity by Vendor
    vendor_orders = df.groupby('Vendor Number')['Order Quantity'].sum().reset_index()
    fig_vendor = px.bar(vendor_orders, x='Vendor Number', y='Order Quantity', title='Order Quantity by Vendor')
    st.plotly_chart(fig_vendor) 


def order_trends_over_time(df, date_col="Pstng Date"):
    """
    Analyzes and visualizes order trends over time.

    Args:
        df: The input DataFrame.
        date_col: The column containing the date (default: 'Pstng Date').
    """
    # Group by date and sum the order quantity
    daily_orders = df.groupby(date_col)['Order Quantity'].sum().reset_index()

    # Create a line chart using Plotly Express
    fig = px.line(daily_orders, x=date_col, y='Order Quantity', title='Order Quantity Over Time')
    st.plotly_chart(fig)

def monthly_order_patterns(df, date_col="Pstng Date"):
    """
    Analyzes and visualizes monthly order patterns.

    Args:
        df: The input DataFrame.
        date_col: The column containing the date (default: 'Pstng Date').
    """
    # Extract month from the date column
    df['Month'] = df[date_col].dt.month_name()

    # Group by month and sum the order quantity
    monthly_orders = df.groupby('Month')['Order Quantity'].sum().reset_index()

    # Create a bar chart using Plotly Express
    fig = px.bar(monthly_orders, x='Month', y='Order Quantity', title='Total Order Quantity by Month')
    st.plotly_chart(fig)

def vendor_material_analysis(df):
    """Analyzes order patterns by vendor and material."""
    vendor_material_orders = df.groupby(['Vendor Number', 'Material Number'])['Order Quantity'].sum().reset_index()
    fig_vendor_material = px.bar(vendor_material_orders, x='Material Number', y='Order Quantity', color='Vendor Number',
                                title='Order Quantity by Vendor and Material')
    st.plotly_chart(fig_vendor_material)

def plant_order_analysis(df):
    """Analyzes order patterns by Plant."""
    plant_orders = df.groupby('Plant')['Order Quantity'].sum().reset_index()
    fig_plant = px.bar(plant_orders, x='Plant', y='Order Quantity', title='Order Quantity by Plant')
    st.plotly_chart(fig_plant)

def purchasing_document_analysis(df):
    """Analyzes order patterns by Purchasing Document."""
    purchasing_orders = df.groupby('Purchasing Document')['Order Quantity'].sum().reset_index()

    # Option 1: Data Table with Sorting
    st.write("**Order Quantity by Purchasing Document:**")
    st.dataframe(purchasing_orders.sort_values(by='Order Quantity', ascending=False))

def order_quantity_distribution(df):
    """Analyzes and visualizes the distribution of order quantities."""
    fig_histogram = px.histogram(df, x='Order Quantity', title='Distribution of Order Quantities')
    st.plotly_chart(fig_histogram)

    fig_box = px.box(df, y='Order Quantity', title='Box Plot of Order Quantities')
    st.plotly_chart(fig_box)

def material_vendor_analysis(df):
    """Analyzes vendor order quantity by material."""
    material_vendor_orders = df.groupby(['Material Number', 'Vendor Number'])['Order Quantity'].sum().reset_index()
    fig = px.bar(material_vendor_orders, x='Material Number', y='Order Quantity', color='Vendor Number', title='Order Quantity by Material and Vendor')
    st.plotly_chart(fig)

def material_plant_analysis(df):
    """Analyzes plant order quantity by material."""
    material_plant_orders = df.groupby(['Material Number', 'Plant'])['Order Quantity'].sum().reset_index()
    fig = px.bar(material_plant_orders, x='Material Number', y='Order Quantity', color='Plant', title='Order Quantity by Material and Plant')
    st.plotly_chart(fig)

def abc_analysis(df):
    """Performs ABC analysis on materials."""
    material_orders = df.groupby('Material Number')['Order Quantity'].sum().reset_index()
    material_orders = material_orders.sort_values(by='Order Quantity', ascending=False)
    material_orders['Cumulative Quantity'] = material_orders['Order Quantity'].cumsum()
    total_quantity = material_orders['Order Quantity'].sum()
    material_orders['Cumulative Percentage'] = (material_orders['Cumulative Quantity'] / total_quantity) * 100
    material_orders['Category'] = pd.cut(material_orders['Cumulative Percentage'], bins=[0, 80, 95, 100], labels=['A', 'B', 'C'])
    st.write("ABC Analysis of Materials:")
    st.dataframe(material_orders)
    # Add legend
    st.markdown("""
    **Legend:**

    * **A:** High-value items. These typically represent the top 80% of the total consumption value.
    * **B:** Medium-value items. These typically represent the next 15% of the total consumption value.
    * **C:** Low-value items. These typically represent the remaining 5% of the total consumption value.
    """)
