import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import norm, poisson, nbinom, gamma, weibull_min, lognorm, expon, beta, kstest, anderson

import plotly.graph_objects as go
import plotly.subplots as sp
import plotly.express as px
import plotly.figure_factory as ff

from utils import *
import consumption_utils
import order_placement_utils
import goods_receipt_utils
import forecast_models
import lead_time_analysis
import DES

# Set the page config with the title centered
st.set_page_config(page_title="Micron SupplySense", layout="wide")

# Center title at the top
st.markdown(
    """
    <h1 style="text-align: center; color: #4B9CD3;">Micron SupplySense</h1>
    """, 
    unsafe_allow_html=True
)

# Create a sidebar for navigation (for a dashboard-style layout)
tabs = st.sidebar.radio("Select an Analysis Type:", ["Material Consumption Analysis", "Order Placement Analysis", "Goods Receipt Analysis","Lead Time Analysis", "Inventory Simulation"])

if tabs == "Material Consumption Analysis":
    st.title("Material Consumption Analysis")
    # Upload Excel files
    uploaded_file = st.file_uploader("Upload Consumption Excel File for Analysis", type=["xlsx"])

    if uploaded_file:
        df = load_data_consumption(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Run analysis functions in the correct order
                df_more_filtered,top_n = consumption_utils.overall_consumption_patterns(df_filtered)
                consumption_utils.outlier_detection(df_more_filtered, top_n)
                consumption_utils.shelf_life_analysis(df_filtered)
                #consumption_utils.vendor_consumption_analysis(df_filtered)
                #consumption_utils.location_consumption_analysis(df_filtered)
                #consumption_utils.batch_variability_analysis(df_filtered)
                #consumption_utils.combined_analysis(df_filtered)
                consumption_utils.specific_material_analysis(df_filtered)

        else:
            st.error("The uploaded file does not contain a 'Material Group' column. Please check the file format.")

elif tabs == "Order Placement Analysis":
    st.title("Order Placement Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload Order Placement Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = order_placement_utils.preprocess_order_data(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Call the analysis functions
                df_more_filtered,top_n = order_placement_utils.overall_orderplacement_patterns(df_filtered)
                order_placement_utils.outlier_detection(df_more_filtered, top_n)
                # order_placement_utils.overall_order_patterns(df_filtered)
                # order_placement_utils.outlier_detection(df_filtered)
                # order_placement_utils.vendor_order_analysis(df_filtered)
                # order_placement_utils.order_trends_over_time(df_filtered)
                # order_placement_utils.monthly_order_patterns(df_filtered)
                # order_placement_utils.vendor_material_analysis(df_filtered)
                # order_placement_utils.plant_order_analysis(df_filtered)
                # order_placement_utils.purchasing_document_analysis(df_filtered)
                # order_placement_utils.order_quantity_distribution(df_filtered)
                # order_placement_utils.material_vendor_analysis(df_filtered)
                # order_placement_utils.supplier_order_analysis(df_filtered)
                # order_placement_utils.material_plant_analysis(df_filtered)
                # order_placement_utils.abc_analysis(df_filtered)
                order_placement_utils.specific_material_analysis(df_filtered)


    else:
        st.write("Please upload an Excel file to begin the analysis.")


elif tabs == "Goods Receipt Analysis":
    st.title("Goods Receipt Analysis")

    # File uploader
    uploaded_file = st.file_uploader("Upload Goods Receipt Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = load_data_GR(uploaded_file)  # Read the file

        # Check if 'Material Group' column exists
        if 'Material Group' in df.columns:
            unique_groups = df['Material Group'].astype(str).unique()  # Get unique values

            for group in unique_groups:
                st.subheader(f"Material Group {group} Analysis")

                # Filter data for the current Material Group
                df_filtered = df[df['Material Group'].astype(str) == str(group)]

                # Call the analysis functions
                df_more_filtered,top_n = goods_receipt_utils.overall_GR_patterns(df_filtered)
                goods_receipt_utils.outlier_detection(df_more_filtered, top_n)
                goods_receipt_utils.specific_material_analysis(df_filtered)


    else:
        st.write("Please upload an Excel file to begin the analysis.")


elif tabs == "Test Page":
    st.title("Forecast Model")

    # File uploader
    uploaded_file = st.file_uploader("Upload Weekly Consumption Data Excel File for Analysis", type="xlsx")
    external_file = st.file_uploader("Upload External Consumption Data Excel File for Analysis", type="xlsx")

    if uploaded_file and external_file:
        df = load_forecast_consumption_data(uploaded_file)  # Read the file
        external_df = load_forecast_consumption_data(external_file)

        # Check if 'Material Number' column exists
        if df is not None and 'Material Number' in df.columns:
            material_numbers = df['Material Number'].unique()
            selected_material_number = st.selectbox("Select Material Number", material_numbers)
            filtered_df = df[df['Material Number'] == selected_material_number].copy() #Make a copy to avoid SettingWithCopyWarning

            model_choice = st.selectbox("Select Model", ["XGBoost", "ARIMA"])
            forecast_weeks = st.number_input("Forecast Weeks", min_value=1, value=6)
            seasonality = st.selectbox("Seasonality", ["Yes", "No"])

            if st.button("Run Forecast"):
                if model_choice == "XGBoost":
                    forecast_results,plt = forecast_models.forecast_weekly_consumption_xgboost_v3(filtered_df, external_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality)
                    st.write("XGBoost Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)
                elif model_choice == "ARIMA":
                    forecast_results,plt = forecast_models.forecast_weekly_consumption_arima_v2(filtered_df, external_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality)
                    st.write("ARIMA Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)

        elif df is not None:
            if 'Material Number' not in df.columns:
                st.error("The uploaded file does not contain a 'Material Number' column.")
            elif 'Week' not in df.columns:
                st.error("The uploaded file does not contain a 'Week' column.")
            elif 'Consumption' not in df.columns:
                st.error("The uploaded file does not contain a 'Consumption' column.")

elif tabs == "Forecast Page V2":
    st.title("Forecast Model")

    # File uploader
    uploaded_file = st.file_uploader("Upload Weekly Consumption Data Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = load_forecast_consumption_data(uploaded_file)  # Read the file

        # Check if 'Material Number' column exists
        if df is not None and 'Material Number' in df.columns:
            material_numbers = df['Material Number'].unique()
            selected_material_number = st.selectbox("Select Material Number", material_numbers)
            filtered_df = df[df['Material Number'] == selected_material_number].copy() #Make a copy to avoid SettingWithCopyWarning

            model_choice = st.selectbox("Select Model", ["XGBoost", "ARIMA"])
            forecast_weeks = st.number_input("Forecast Weeks", min_value=1, value=6)
            seasonality = st.selectbox("Seasonality", ["Yes", "No"])

            if st.button("Run Forecast"):
                if model_choice == "XGBoost":
                    forecast_results,plt = forecast_models.forecast_weekly_consumption_xgboost(filtered_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality)
                    st.write("XGBoost Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)
                elif model_choice == "ARIMA":
                    forecast_results,plt = forecast_models.forecast_weekly_consumption_arima(filtered_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality)
                    st.write("ARIMA Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)

        elif df is not None:
            if 'Material Number' not in df.columns:
                st.error("The uploaded file does not contain a 'Material Number' column.")
            elif 'Week' not in df.columns:
                st.error("The uploaded file does not contain a 'Week' column.")
            elif 'Consumption' not in df.columns:
                st.error("The uploaded file does not contain a 'Consumption' column.")

if tabs == "Forecast Page":
    st.title("Forecast Model")

    # File uploader
    uploaded_file = st.file_uploader("Upload Weekly Consumption Data Excel File for Analysis", type="xlsx")

    if uploaded_file:
        df = load_forecast_consumption_data(uploaded_file)  # Read the file

        # Check if 'Material Number' column exists
        if df is not None and 'Material Number' in df.columns:
            material_numbers = df['Material Number'].unique()
            selected_material_number = st.selectbox("Select Material Number", material_numbers)
            filtered_df = df[df['Material Number'] == selected_material_number].copy()  # Make a copy

            model_choice = st.selectbox("Select Model", ["XGBoost", "ARIMA"])
            forecast_weeks = 6  # Fixed to 6 weeks
            seasonality = "Yes"  # Fixed to Yes

            if st.button("Run Forecast"):
                if model_choice == "XGBoost":
                    forecast_results, plt = forecast_models.forecast_weekly_consumption_xgboost(
                        filtered_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality
                    )
                    st.write("XGBoost Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)
                elif model_choice == "ARIMA":
                    forecast_results, plt = forecast_models.forecast_weekly_consumption_arima(
                        filtered_df, forecast_weeks_ahead=forecast_weeks, seasonality=seasonality
                    )
                    st.write("ARIMA Forecast Results:")
                    st.pyplot(plt)
                    st.write(forecast_results)

        elif df is not None:
            if 'Material Number' not in df.columns:
                st.error("The uploaded file does not contain a 'Material Number' column.")
            elif 'Week' not in df.columns:
                st.error("The uploaded file does not contain a 'Week' column.")
            elif 'Consumption' not in df.columns:
                st.error("The uploaded file does not contain a 'Consumption' column.")

elif tabs == "Lead Time Analysis":
    st.title("")

    # File uploader
    uploaded_file_op = st.file_uploader("Upload Order Placement Excel File for Analysis", type="xlsx")
    uploaded_file_gr = st.file_uploader("Upload Goods Received Excel File for Analysis", type="xlsx")
    uploaded_file_sr = st.file_uploader("Upload Modified Shortage Report Excel File for Analysis", type="xlsx")

    if uploaded_file_op and uploaded_file_gr and uploaded_file_sr:
        with st.spinner("Processing lead time analysis..."):
            op_df = pd.read_excel(uploaded_file_op)
            gr_df = pd.read_excel(uploaded_file_gr)
            shortage_df = pd.read_excel(uploaded_file_sr)

            matched, unmatched_op, unmatched_gr = lead_time_analysis.process_dataframes(op_df, gr_df)
            calculated_df = lead_time_analysis.calculate_actual_lead_time(matched)
            final_df = lead_time_analysis.calculate_lead_time_summary(shortage_df)
            final_result = lead_time_analysis.calculate_lead_time_differences(final_df, calculated_df)

            # Call the updated Plotly version of your function
            fig1, fig2, fig3, fig4 = lead_time_analysis.analyze_and_plot_lead_time_differences_plotly(final_result)

        st.success("Lead Time Analysis Completed ✅")
        st.write("### Lead Time Analysis Results:")
        st.plotly_chart(fig1, use_container_width=True)
        st.plotly_chart(fig2, use_container_width=True)
        st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.write("Please upload all Excel files to begin the analysis.")

            
elif tabs == "Inventory Simulation":
    st.title("Inventory Simulation")

    # File uploader
    uploaded_consumption = st.file_uploader("Upload Consumption File", type="xlsx")
    uploaded_goods_receipt = st.file_uploader("Upload Goods Receipt Excel File", type="xlsx")
    uploaded_order_placement = st.file_uploader("Upload Order Placement Excel File", type="xlsx")
    uploaded_merged = st.file_uploader("Upload Merged Shortage Excel File", type="xlsx")

    # Load files only when they are first uploaded
    DES.load_and_store_file(uploaded_consumption, "consumption_df")
    DES.load_and_store_file(uploaded_goods_receipt, "gr_df")
    DES.load_and_store_file(uploaded_order_placement, "order_df")
    DES.load_and_store_file(uploaded_merged, "merged_df")

    # Access stored files without reloading
    if "consumption_df" in st.session_state:
        consumption_df = st.session_state["consumption_df"]

    if "gr_df" in st.session_state:
        gr_df = st.session_state["gr_df"]

    if "order_df" in st.session_state:
        order_df = st.session_state["order_df"]

    if "merged_df" in st.session_state:
        merged_df = st.session_state["merged_df"]

    if "consumption_df" in st.session_state and "order_df" in st.session_state and "gr_df" in st.session_state and "merged_df" in st.session_state:
        # consumption_df = DES.load_data(uploaded_consumption)
        consumption_df = DES.preprocess_data_consumption(consumption_df)
        # gr_df = DES.load_data(uploaded_goods_receipt)
        gr_df = DES.preprocess_data_GR(gr_df)
        # order_df = DES.load_data(uploaded_order_placement)
        order_df = DES.preprocess_data_OP(order_df)
        # merged_df = DES.load_data(uploaded_merged)

        col1, col2, col3 = st.columns(3)

        # Filtering Data Before Running Simulation
        with col1:
            selected_material = st.selectbox("Select Material", consumption_df['Material Number'].unique())
            
        with col2:
            selected_plant = st.selectbox("Select Plant", consumption_df['Plant'].unique())

        with col3:
            selected_site = st.selectbox("Select Site", consumption_df['Site'].unique())

        filtered_consumption = consumption_df[(consumption_df['Material Number'] == selected_material) & 
                                        (consumption_df['Plant'] == selected_plant) & 
                                        (consumption_df['Site'] == selected_site)]
        filtered_orders = order_df[(order_df['Material Number'] == selected_material) & 
                                (order_df['Plant'] == selected_plant)]
        filtered_receipts = gr_df[(gr_df['Material Number'] == selected_material) & 
                                    (gr_df['Plant'] == selected_plant) & 
                                    (gr_df['Site'] == selected_site)]
        filtered_merged = merged_df[(merged_df['Material Number'] == selected_material) & 
                                    (merged_df['Plant'] == selected_plant) & 
                                    (merged_df['Site'] == selected_site) & (merged_df['Measures'] == 'Supply')]
        
        max_lead_time, std_lead_time, dist_name, dist_params = DES.process_lead_time(filtered_merged)

        with col1:
            num_weeks = st.number_input("Number of Simulation Weeks", min_value=1, value=52)
            st.info("Set the number of weeks for the simulation.")

            initial_inventory = st.number_input("Initial Inventory", min_value=10, max_value=20000, value=50)
            st.info("The starting inventory level for the simulation.")

            # Consumption Input
            consumption_type = st.radio("Consumption Type", ["Fixed", "Distribution"])
            consumption_values = []
            consumption_distribution_params = None  # Initialize to avoid potential errors later
            safety_stock = 0
            if consumption_type == "Fixed":
                fixed_consumption = st.number_input("Fixed Consumption Value", min_value=0, value=10)
                consumption_values = [fixed_consumption] * num_weeks

                service_level_percentage = st.number_input("Desired Service Level (%)", min_value=1, max_value=100, value=95)
                service_level = service_level_percentage / 100.0
                std_dlt = np.sqrt(max_lead_time) * np.std(consumption_values)
                z_score = stats.norm.ppf(service_level)
                safety_stock = z_score * std_dlt

                st.success(f"Calculated Safety Stock: {safety_stock:.2f} units")

                average_consumption = np.mean(consumption_values)
                lead_time_demand = average_consumption * max_lead_time

                # 2. Calculate Reorder Point
                reorder_pt_calc = lead_time_demand + safety_stock

                reorder_point = st.number_input("Reorder Point", min_value=5, max_value=500, value=int(reorder_pt_calc))
                st.info("The inventory level at which a new order is placed.")

            else:  # Consumption Type is "Distribution"
                consumption_values = filtered_consumption.iloc[:, 3:].values.flatten()
                consumption_distribution_params  = DES.fit_distribution(consumption_values, "Consumption")

                mean_consumption = DES.get_mean_from_distribution(consumption_distribution_params)
                std_consumption = DES.get_std_from_distribution(consumption_distribution_params)
                if consumption_distribution_params:
                    simulated_demand = DES.simulate_demand(consumption_distribution_params)
                    lead_time_values = filtered_merged.filter(like="Lead Time").iloc[0].dropna().astype(float)
                    lead_time_distribution_params  = DES.fit_distribution(lead_time_values, "Lead Time")
                    simulated_lead_times = DES.simulate_demand(lead_time_distribution_params)

                service_level_percentage = st.number_input("Desired Service Level (%)", min_value=1, max_value=100, value=95)
                service_level = service_level_percentage / 100.0
                # std_dlt = np.sqrt(max_lead_time) * std_consumption
                # z_score = stats.norm.ppf(service_level)
                # safety_stock = z_score * std_dlt

                simulated_stock_levels = simulated_lead_times * simulated_demand
                # Calculate safety stock as the percentile of the simulated stock levels based on the service level
                safety_stock = np.percentile(simulated_stock_levels, service_level)

                st.success(f"Calculated Safety Stock: {safety_stock:.2f} units")

                lead_time_demand = mean_consumption * max_lead_time

                # 2. Calculate Reorder Point
                reorder_pt_calc = lead_time_demand + safety_stock
                try:
                    reorder_point = st.number_input("Reorder Point", min_value=5, max_value=500, value=int(reorder_pt_calc))
                except Exception:
                    reorder_point = st.number_input("Reorder Point", min_value=5, max_value=500, value=100)
                st.info("The inventory level at which a new order is placed.")

        with col2:
            lead_time = st.number_input("Lead Time (weeks)", min_value=1.0, max_value=20.0, value=float(max_lead_time))
            st.info("The time (in weeks) it takes for an order to arrive after it is placed.")

            # Demand Surge Controls
            demand_surge_weeks_options = [f"WW{i+1}" for i in range(num_weeks)]
            demand_surge_weeks_input = st.multiselect("Demand Surge Weeks", demand_surge_weeks_options)
            st.info("Select the weeks where you want to simulate a sudden increase in demand.")

            # Order Quantity Input
            order_quantity_type = st.radio("Order Quantity Type", ["Fixed", "Distribution"])
            order_quantity = 0
            order_distribution_params = None

            if order_quantity_type == "Fixed":
                order_quantity = st.number_input("Order Quantity", min_value=10, max_value=10000, value=50)
            else:  # Order Quantity Type is "Distribution"
                order_values = filtered_orders.iloc[:, 3:].values.flatten()
                order_distribution_params = DES.fit_distribution(order_values, "Order Quantity")
                

        with col3:
            lead_time_std_dev = st.number_input("Lead Time Std Dev (weeks)", min_value=0.0, max_value=10.0, value=float(std_lead_time))
            st.info("The standard deviation of the lead time, representing variability.")
            
            demand_surge_factor = st.number_input("Demand Surge Factor", min_value=0.5, max_value=5.0, value=2.0, step=0.1)
            st.info("Enter the factor by which demand will increase during the selected weeks. (e.g., 2.0 doubles demand)")

            N = st.number_input("Number of Monte Carlo Simulations", min_value=1, max_value=10000, value=100)
            st.info("The number of Monte Carlo simulations to run. A higher number provides more accurate results but requires more computation.")

        if st.button("Run Simulation"):
            with st.spinner("Running simulation..."):
                args = (filtered_consumption, filtered_orders, filtered_receipts, initial_inventory, reorder_point, order_quantity, lead_time, lead_time_std_dev, demand_surge_weeks_input, demand_surge_factor, consumption_distribution_params, consumption_type, consumption_values, num_weeks, order_distribution_params, order_quantity_type)

                # Run Monte Carlo simulation
                
                all_inventory_histories, all_proactive_inventory_histories, all_stockout_weeks, all_proactive_stockout_weeks, all_wos_histories, all_proactive_wos_histories, all_consumption_histories, all_weekly_events = DES.run_monte_carlo_simulation(N, *args)
                # Compute averages
            
                avg_inventory, avg_wos, avg_consumption, stockout_frequency, avg_proactive_inventory,avg_proactive_wos, stockout_frequency_proactive = DES.compute_averages(all_inventory_histories, all_proactive_inventory_histories, all_stockout_weeks, all_proactive_stockout_weeks, all_wos_histories, all_proactive_wos_histories, all_consumption_histories)

                # # Find the representative run
                representative_index = DES.find_representative_run(all_inventory_histories, avg_inventory)

                # # Get details of the representative run
                representative_inventory, representative_inventory_proactive, representative_stockout_weeks, representative_stockout_weeks_proactive, representative_wos, representative_wos_proactive, representative_consumption, representative_weekly_events = DES.get_representative_run_details(representative_index, all_inventory_histories, all_proactive_inventory_histories, all_stockout_weeks, all_proactive_stockout_weeks, all_wos_histories, all_proactive_wos_histories, all_consumption_histories, all_weekly_events)

                # Monte Carlo Simulation (Proactive + Reactive)

                week_numbers = list(range(1, num_weeks + 1))
                # Create DataFrames for inventory, WoS, and consumption
                inventory_df = pd.DataFrame({
                    'Working Week': week_numbers,
                    'Reactive Inventory': representative_inventory,
                    'Proactive Inventory': representative_inventory_proactive
                })

                wos_df = pd.DataFrame({
                    'Working Week': week_numbers,
                    'Reactive WoS': representative_wos,
                    'Proactive WoS': representative_wos_proactive
                })

                consumption_df = pd.DataFrame({
                    'Working Week': week_numbers,
                    'Consumption': representative_consumption
                })

                # Visualization
                fig_inventory = px.line(inventory_df, x='Working Week', y=['Reactive Inventory', 'Proactive Inventory'], title='Inventory Over Time')
                fig_inventory.update_xaxes(dtick=5)
                fig_inventory.update_layout(yaxis_title='Inventory')

                # Highlight stockout weeks
                if representative_stockout_weeks:
                    fig_inventory.add_vrect(x0=min(representative_stockout_weeks), x1=max(representative_stockout_weeks), fillcolor="red", opacity=0.25, line_width=0, annotation_text="Stockout Weeks", annotation_position="top left")
                if representative_stockout_weeks_proactive:
                    fig_inventory.add_vrect(x0=min(representative_stockout_weeks_proactive), x1=max(representative_stockout_weeks_proactive), fillcolor="orange", opacity=0.25, line_width=0, annotation_text="Proactive Stockout Weeks", annotation_position="top right")

                st.plotly_chart(fig_inventory)

                fig_wos = px.line(wos_df, x='Working Week', y=['Reactive WoS', 'Proactive WoS'], title='Weeks of Supply (WoS) Over Time')
                fig_wos.update_xaxes(dtick=5)
                fig_wos.update_layout(yaxis_title='Weeks of Supply')

                # Highlight stockout weeks
                if representative_stockout_weeks:
                    fig_wos.add_vrect(x0=min(representative_stockout_weeks), x1=max(representative_stockout_weeks), fillcolor="red", opacity=0.25, line_width=0, annotation_text="Stockout Weeks", annotation_position="top left")
                if representative_stockout_weeks_proactive:
                    fig_wos.add_vrect(x0=min(representative_stockout_weeks_proactive), x1=max(representative_stockout_weeks_proactive), fillcolor="orange", opacity=0.25, line_width=0, annotation_text="Proactive Stockout Weeks", annotation_position="top right")

                st.plotly_chart(fig_wos)

                fig_consumption = px.line(consumption_df, x='Working Week', y='Consumption', title='Consumption Over Time')
                fig_consumption.update_xaxes(dtick=5)
                fig_consumption.update_layout(yaxis_title='Consumption')

                st.plotly_chart(fig_consumption)

                # Display stockout information
                if representative_stockout_weeks:
                    st.warning(f"Reactive stockout occurred in weeks: {', '.join(map(str, representative_stockout_weeks))}")
                else:
                    st.success("No reactive stockouts occurred.")

                if representative_stockout_weeks_proactive:
                    st.warning(f"Proactive stockout occurred in weeks: {', '.join(map(str, representative_stockout_weeks_proactive))}")
                else:
                    st.success("No proactive stockouts occurred.")

                st.subheader("Weekly Simulation Events after Monte Carlo")
                for event in representative_weekly_events:
                    st.markdown(event)