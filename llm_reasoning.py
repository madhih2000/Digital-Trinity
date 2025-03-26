from groq import Groq
import streamlit as st
import pandas as pd

API_KEY = st.secrets["groq"]["API_KEY"]


def explain_box_plot_with_groq_consumption(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical consumption.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
        )

        explanation = chat_completion.choices[0].message.content
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during Groq API call: {e}")

def explain_box_plot_with_groq_orderplacement(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Order Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Order Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical order placement.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Order Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
        )

        explanation = chat_completion.choices[0].message.content
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during Groq API call: {e}")

def explain_box_plot_with_groq_goods_receipt(df, material_column="Material Number"):
    """
    Explains boxplot or variance of a DataFrame column using Groq LLM.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        column_name (str): The name of the column to explain.
        api_key (str): Your Groq API key.
    """

    # Calculate statistics using groupby and agg
    material_stats = df.groupby(material_column)['Quantity'].agg(
        min='min',
        max='max',
        median='median',
        q1=lambda x: x.quantile(0.25),
        q3=lambda x: x.quantile(0.75),
        percentile_90=lambda x: x.quantile(0.90)
    ).reset_index()

    # Calculate outliers (IQR method)
    def calculate_outliers(group):
        q1 = group.quantile(0.25)
        q3 = group.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return list(group[(group < lower_bound) | (group > upper_bound)])

    outliers = df.groupby(material_column)['Quantity'].apply(calculate_outliers).reset_index(name='outliers')

    # Merge outliers with stats
    material_stats = pd.merge(material_stats, outliers, on=material_column)

    # Create a string representation of the statistics
    stats_string = material_stats.to_string(index=False)

    client = Groq(api_key=API_KEY)

    data_description = f"Boxplot data from materials. Quantity data statistics:\n{stats_string}"

    system_prompt = """
    You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
    Your task is to interpret a string description of boxplot statistics representing material quantities at the material number level which are from past historical goods receipt data.
    Provide key insights and interpretations of the boxplot data in bullet points.
    Focus on:
    - Compare and contrast the distribution of 'Quantity' across different material numbers.
    - Highlight significant differences in variance (spread) and skewness.
    - Identify and interpret the impact of notable outliers.
    - Summarize key trends and patterns observed in the data.
    - Integrate into your analysis the understanding that while the box plots are sorted by largest variance, visual inspection of the box plot's spread, interquartile range, or absolute differences between medians can be misleading when trying to directly infer variance. Variance is a statistical measure of the average squared deviation from the mean, and should be calculated directly from the data. Do not simply state this explanation as a separate point; weave it into your analysis.
    - Do not simply repeat the statistical values. Provide analytical interpretations.
    Do not include any introductory phrases or preambles. Start directly with the bullet points.
    """

    user_prompt = f"""
    Explain the boxplot for the following data:\n\n{data_description}
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
        )

        explanation = chat_completion.choices[0].message.content
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during Groq API call: {e}")

def explain_inventory_events(representative_weekly_events, reorder_point, lead_time, lead_time_std_dev, consumption_distribution_params, consumption_type,consumption_best_distribution, order_distribution_params, order_quantity_type, order_distribution_best):
    """
    Explains weekly inventory events in terms of recommended inventory policies to prevent stockout.

    Args:
        representative_weekly_events (list): A list of strings representing the weekly events.
        reorder_point (int): The reorder point used in the simulation.
        lead_time (int): The average lead time used in the simulation.
        lead_time_std_dev (float): The standard deviation of the lead time.
        consumption_distribution_params (dict): Parameters of the consumption distribution, if applicable.
        consumption_type (str): "Fixed" or "Distribution".
        order_distribution_params (dict): Parameters of the order quantity distribution, if applicable.
        order_quantity_type (str): "Fixed" or "Distribution".
    """

    system_prompt = f"""
    You are an expert supply chain analyst specializing in inventory management with extensive experience in analyzing inventory events and recommending policies to prevent stockout. Your role is to analyze the weekly inventory events and provide actionable insights based on your findings.
    Your task is to interpret the weekly inventory events and recommend inventory policies to prevent stockout. 
     Analyze the provided weekly inventory events, considering:
        * Identify Stockout/Near-Stockout Events:
            * Analyze the provided weekly inventory events to pinpoint specific weeks where stockout or near-stockout situations occurred, considering both reactive and proactive inventory levels.
            * Specifically, examine weeks where ending inventory levels dropped significantly or approached zero.
            * Use the provided data to quantify the severity of these events.
            * Proactively suggest potential root causes for these events, based on the provided data. Explain the root cause in detail using the data provided.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Analyze Consumption Patterns:
            * Detail and explain the consumption patterns in the identified critical weeks.
            * Correlate consumption with the provided distribution source (probability distribution identified) and assess if there are any significant deviations from the expected behavior.
            * Calculate and present the variance in consumption week to week.
            * Quantify the difference between forcasted and actual consumption. Actual consumption is the simulated consumption based on the fixed or distribution identified and forecasted consumption is using the forecasting models. 
            * Consider that consumption type is: {consumption_type}. If consumption type is Distribution, the distribution parameters were: {consumption_distribution_params} and the probability distribution it follows is {consumption_best_distribution}. Explain how the distribution parameters affected the consumption, if applicable.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Evaluate Inventory Levels:
            * Compare reactive and proactive inventory levels in the critical weeks.
            * Determine the impact of reactive and proactive orders on preventing or mitigating stockouts.
            * Calculate the time between order and arrival for both reactive and proactive orders.
            * Consider the lead time was {lead_time} with a standard deviation of {lead_time_std_dev}.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Proactive vs. Reactive Inventory:
            * Based on the data, explicitly demonstrate the advantages of proactive inventory ordering over reactive ordering.
            * Quantify the differences in inventory levels and stockout occurrences between the two strategies.
            * Explain how the provided forcasted consumption data could have been better used.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Inventory Policy Recommendations:
            * Provide specific, data-driven recommendations for inventory policies to prevent future stockouts for the given sequence of events.
            * Suggest optimal reorder points, safety stock levels, and order quantities for both reactive and proactive inventory for the given data or how the order placement should have been done better.
            * Recommend adjustments to the forecasting method based on the observed consumption patterns.
            * Recommend a policy that accounts for the variation in the consumption.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Ordering Strategy Improvements:
            * Analyze the current ordering strategy and identify its weaknesses.
            * Suggest improvements to the timing and frequency of orders, considering the lead times and consumption variability.
            * Recommend if the reorder point should be changed, and by how much. The reorder point used was {reorder_point}.
            * Consider that order quantity type is: {order_quantity_type}. If order quantity type is Distribution, the distribution parameters were: {order_distribution_params} and the distribution it followed is {order_distribution_best}. Explain how the order distribution parameters affected the order quantity, if applicable.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Summary of Key Insights and Recommendations:
            * Provide a concise summary of the key insights derived from the data.
            * Present a clear and actionable list of recommendations to optimize inventory management and prevent future stockouts.
            * Include the calculated variance of consumption, and the calculated lead times, and the calculated difference between forecasted and actual consumption in the summary.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
    Do not include any introductory phrases or preambles. 
    Each section should have multiple bullet points, and each bullet point may contain more than one sentence where necessary to ensure clarity and depth in the analysis.
    Use the data provided to pinpoint specific weeks and events that led to stockout. Provide detailed, data-driven insights.
    """
    weekly_events_text = "\n\n".join(representative_weekly_events)

    user_prompt = f"""
    Explain the weekly inventory events and recommend inventory policies to prevent stockout or excess inventory:

    {weekly_events_text}
    """

    try:
        client = Groq(api_key=API_KEY)

        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model="llama-3.3-70b-versatile",
        )

        explanation = chat_completion.choices[0].message.content
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during Groq API call: {e}")