import streamlit as st
import pandas as pd
import numpy as np
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
tabs = st.sidebar.radio("Select an Analysis Type:", ["Material Consumption Analysis", "Order Placement Analysis", "Goods Receipt Analysis","Lead Time Analysis"])

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

            
