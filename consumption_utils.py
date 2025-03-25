import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import llm_reasoning

def overall_consumption_patterns(df, material_column='Material Number'):
    """
    Analyzes and visualizes overall consumption patterns, applying a common set of filters to both graphs.
    Top N filter works differently for transaction and consumption graphs.
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
    # GRAPH 2 - Overall Consumption
    # -------------------------------------------------------------------

    # Use the globally filtered DataFrame
    #df_consumption = df_filtered.copy()

    # Top N Filtering for Consumption
    #Assumes there is a column named 'Quantity'
    material_consumption_sum = df_transactions.groupby(material_column)['Quantity'].sum().abs().reset_index() #Absolute sum
    material_consumption_sum = material_consumption_sum.sort_values(by='Quantity', ascending=False)

    if top_n != 'All':
        top_n_int = int(top_n) #To convert from string to int
        top_materials_cons = material_consumption_sum[material_column].head(top_n_int).tolist()
        df_transactions = df_transactions[df_transactions[material_column].isin(top_materials_cons)]

    # Visualization - Overall Consumption
    fig_overall = px.bar(material_consumption_sum, x=material_column, y='Quantity',
                         title=f'Overall Consumption by {material_column}')
    st.plotly_chart(fig_overall)


    # -------------------------------------------------------------------
    # Data Display (For Debugging Purposes)
    # -------------------------------------------------------------------

    #st.write("Final Dataframe after Global Filters:")
    #st.dataframe(df_filtered)

    return df_filtered, top_n #Important, must return for the following code to work


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

    llm_reasoning.explain_box_plot_with_groq_consumption(df_filtered)

def specific_material_analysis(df, material_column='Material Number'):
    """
    Analyzes consumption patterns for a specific material, including consumption trend, seasonal subseries,
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
    # Time Series of Consumption (Trend)
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
        title=f'Consumption Trend and Transaction Count ({aggregation_level}) for {selected_material}',
        xaxis_title='Date',
        yaxis_title='Quantity',
        yaxis2=dict(
            title='Transaction Count',
            overlaying='y',
            side='right'
        )
    )

    st.plotly_chart(fig)

    # # Convert 'Pstng Date' to datetime objects
    # try:
    #     filtered_material_data['Pstng Date'] = pd.to_datetime(filtered_material_data['Pstng Date'], format='%d/%m/%Y %I:%M:%S %p', errors='raise')
    # except (ValueError, KeyError) as e:
    #     st.error(f"Error converting 'Pstng Date' to datetime: {e}. Ensure the column exists and contains valid date values.")
    #     return

    # min_date = filtered_material_data['Pstng Date'].min().date()
    # max_date = filtered_material_data['Pstng Date'].max().date()

    # start_date, end_date = st.date_input("Select Date Range",
    #                                     [min_date,
    #                                      max_date])

    # # Create a date range DataFrame to ensure all dates are present
    # date_range = pd.date_range(start=start_date, end=end_date)
    # date_df = pd.DataFrame({'Pstng Date': date_range})

    # filtered_time_data = filtered_material_data[(filtered_material_data['Pstng Date'].dt.date >= start_date) &
    #                                             (filtered_material_data['Pstng Date'].dt.date <= end_date)]

    # aggregation_level = st.selectbox("Select Aggregation Level", ['Daily', 'Weekly', 'Monthly', 'Quarterly'])

    # if aggregation_level == 'Daily':
    #     aggregated_data = filtered_time_data.groupby('Pstng Date')['Quantity'].sum().reset_index()
    #     # Merge with date range to fill missing dates with 0
    #     aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
    #     aggregated_data = pd.merge(date_df, aggregated_data, on='Pstng Date', how='left').fillna(0)

    # elif aggregation_level == 'Weekly':
    #     aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W'))['Quantity'].sum().reset_index()
    #     # Create a weekly date range to merge with
    #     date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
    #     date_df_weekly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='W')).min().reset_index() # Get the first day of the week
    #     aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
    #     aggregated_data = pd.merge(date_df_weekly, aggregated_data, on='Pstng Date', how='left').fillna(0) #Merge on the first day of the week

    # elif aggregation_level == 'Monthly':
    #     aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M'))['Quantity'].sum().reset_index()
    #     # Create monthly date range
    #     date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
    #     date_df_monthly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='M')).min().reset_index() # Get the first day of the month.
    #     aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
    #     aggregated_data = pd.merge(date_df_monthly, aggregated_data, on='Pstng Date', how='left').fillna(0)  # Merge on the first day of the month
    # elif aggregation_level == 'Quarterly':
    #     aggregated_data = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q'))['Quantity'].sum().reset_index()
    #     #Create quarterly data range
    #     date_df['Pstng Date'] = pd.to_datetime(date_df['Pstng Date'])
    #     date_df_quarterly = date_df.groupby(pd.Grouper(key='Pstng Date', freq='Q')).min().reset_index()
    #     aggregated_data['Pstng Date'] = pd.to_datetime(aggregated_data['Pstng Date'])
    #     aggregated_data = pd.merge(date_df_quarterly, aggregated_data, on = 'Pstng Date', how = 'left').fillna(0)

    # fig_time_series = px.line(aggregated_data, x='Pstng Date', y='Quantity',
    #                          title=f'Consumption Trend ({aggregation_level}) for {selected_material}')
    # st.plotly_chart(fig_time_series)

    # # -------------------------------------------------------------------
    # # Transaction Count Over Time
    # # -------------------------------------------------------------------
    # st.subheader("Transaction Count Over Time")

    # if aggregation_level == 'Daily':
    #     transaction_counts = filtered_time_data.groupby(filtered_time_data['Pstng Date'].dt.date).size().reset_index(name='Transaction Count')
    #     transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
    #     transaction_counts = pd.merge(date_df, transaction_counts, on='Pstng Date', how='left').fillna(0)
    # elif aggregation_level == 'Weekly':
    #     transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='W')).size().reset_index(name='Transaction Count')
    #     transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
    #     transaction_counts = pd.merge(date_df_weekly, transaction_counts, on='Pstng Date', how='left').fillna(0)
    # elif aggregation_level == 'Monthly':
    #     transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='M')).size().reset_index(name='Transaction Count')
    #     transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
    #     transaction_counts = pd.merge(date_df_monthly, transaction_counts, on='Pstng Date', how='left').fillna(0)
    # elif aggregation_level == 'Quarterly':
    #     transaction_counts = filtered_time_data.groupby(pd.Grouper(key='Pstng Date', freq='Q')).size().reset_index(name='Transaction Count')
    #     transaction_counts['Pstng Date'] = pd.to_datetime(transaction_counts['Pstng Date'])
    #     transaction_counts = pd.merge(date_df_quarterly, transaction_counts, on='Pstng Date', how='left').fillna(0)

    # fig_transactions = px.line(transaction_counts, x='Pstng Date', y='Transaction Count',
    #                             title=f'Number of Transactions Over Time ({aggregation_level}) for {selected_material}')
    # st.plotly_chart(fig_transactions)

    # -------------------------------------------------------------------
    # Seasonal Subseries Plot (Monthly)
    # -------------------------------------------------------------------

    #st.subheader("Seasonal Subseries Plot (Monthly)") # Add a section title
    '''
    if aggregation_level in ['Monthly', 'Quarterly']:
        filtered_time_data['Month'] = filtered_time_data['Pstng Date'].dt.month
        fig_seasonal_subseries = px.box(filtered_time_data, x='Month', y='Quantity',
                                       title=f'Seasonal Subseries Plot (Monthly) for {selected_material}')
        st.plotly_chart(fig_seasonal_subseries)
    '''

def shelf_life_analysis(df):
    """
    Analyzes and visualizes shelf life, separating finite and infinite shelf life items
    based on the assumption that items with '2100-01-01' in 'SLED/BBD' have infinite shelf life.

    Args:
        df: The input DataFrame.
    """
    df['Remaining Shelf Life (Days)'] = (df['SLED/BBD'] - df['Pstng Date']).dt.days

    # Identify infinite shelf life items (those with '2100-01-01' in 'SLED/BBD')
    infinite_shelf_life = df[df['SLED/BBD'] == pd.to_datetime('2100-01-01')]
    finite_shelf_life = df[df['SLED/BBD'] != pd.to_datetime('2100-01-01')]

    # Change 'SLED/BBD' to 'No Expiry' for infinite shelf life items
    infinite_shelf_life['SLED/BBD'] = 'No Expiry'

    # Drop 'Remaining Shelf Life (Days)' column from infinite_shelf_life
    infinite_shelf_life = infinite_shelf_life.drop(columns=['Remaining Shelf Life (Days)'], errors='ignore')

    # Visualization for finite shelf life
    if not finite_shelf_life.empty:
        fig_hist_finite = px.histogram(finite_shelf_life, x='Remaining Shelf Life (Days)',
                                        title='Distribution of Remaining Shelf Life (Days) - Finite Shelf Life')
        st.plotly_chart(fig_hist_finite)
    else:
        st.write("No items with finite shelf life in the selected data.")

    infinite_shelf_life = infinite_shelf_life.reset_index(drop = True)
    # Count of infinite shelf life items
    st.write(f"Number of Items with Infinite Shelf Life: {len(infinite_shelf_life)}")

    # Display a table with infinite shelf life items
    if not infinite_shelf_life.empty:
        st.write("Items with Infinite Shelf Life:")
        st.write(infinite_shelf_life)
    else:
        st.write("No items with infinite shelf life in the selected data.")


def vendor_consumption_analysis(df):
    """
    Analyzes consumption patterns for each vendor.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Vendor
    vendor_consumption = df.groupby('Vendor Number')['Quantity'].sum().reset_index()
    fig_vendor = px.bar(vendor_consumption, x='Vendor Number', y='Quantity', title='Consumption by Vendor')
    st.plotly_chart(fig_vendor)

    # Time Series of Consumption by Vendor (Example with Monthly Aggregation)
    monthly_vendor_consumption = df.groupby(['Vendor Number', pd.Grouper(key='Pstng Date', freq='M')])['Quantity'].sum().reset_index()
    fig_vendor_time_series = px.line(monthly_vendor_consumption, x='Pstng Date', y='Quantity', color='Vendor Number',
                                     title='Monthly Consumption Trend by Vendor')
    st.plotly_chart(fig_vendor_time_series)


def location_consumption_analysis(df):
    """
    Analyzes consumption patterns at the plant and site level.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Plant
    plant_consumption = df.groupby('Plant')['Quantity'].sum().reset_index()
    fig_plant = px.bar(plant_consumption, x='Plant', y='Quantity', title='Consumption by Plant')
    st.plotly_chart(fig_plant)

    # Consumption by Site
    site_consumption = df.groupby('Site')['Quantity'].sum().reset_index()
    fig_site = px.bar(site_consumption, x='Site', y='Quantity', title='Consumption by Site')
    st.plotly_chart(fig_site)


def batch_variability_analysis(df):
    """
    Analyzes the variability in consumption across different batches of the same material.

    Args:
        df: The input DataFrame.
    """
    # Calculate consumption by Material Number and Batch
    batch_consumption = df.groupby(['Material Number', 'Batch'])['Quantity'].sum().reset_index()

    # Calculate variance of consumption for each material across its batches
    batch_variance = batch_consumption.groupby('Material Number')['Quantity'].var().reset_index(name='Consumption Variance')

    # Visualize variance using a bar chart
    fig_variance = px.bar(batch_variance, x='Material Number', y='Consumption Variance',
                          title='Variance of Consumption Across Batches')
    st.plotly_chart(fig_variance)


def combined_analysis(df):
    """
    Performs combined analysis of consumption patterns by vendor and material number,
    site and material number, and batch and site.

    Args:
        df: The input DataFrame.
    """
    # Consumption by Vendor and Material Number
    vendor_material_consumption = df.groupby(['Vendor Number', 'Material Number'])['Quantity'].sum().reset_index()
    fig_vendor_material = px.bar(vendor_material_consumption, x='Material Number', y='Quantity', color='Vendor Number',
                                 title='Consumption by Vendor and Material Number')
    st.plotly_chart(fig_vendor_material)

    # Consumption by Site and Material Number
    site_material_consumption = df.groupby(['Site', 'Material Number'])['Quantity'].sum().reset_index()
    fig_site_material = px.bar(site_material_consumption, x='Material Number', y='Quantity', color='Site',
                               title='Consumption by Site and Material Number')
    st.plotly_chart(fig_site_material)
