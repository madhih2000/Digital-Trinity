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

def explain_waterfall_chart_with_groq(df):
    """
    Explains the root cause analysis of a waterfall chart.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        material_column (str): The name of the column containing material numbers.
    """
    df_string = df.to_string(index=False)
    client = Groq(api_key=API_KEY)

    system_prompt = """
            You are an expert supply chain analyst specializing in the semiconductor industry with extensive experience in data analysis and interpretation. Your role is to analyze statistical data and provide actionable insights based on your findings.
            
            Your task is to interpret a string description of waterfall chart dataframe representing supply chain values at the material number level which are from past historical data. The data is structured in a weekly format labelled as “WW” and the respective week number.
            
            Provide key insights and interpretations of the waterfall chart data in bullet points.
            
            Focus on:
            1.  Identify any negative values in the rows 'Weeks of Stock', and highlight if there are significant differences of more than 4 weeks. 
            2.  Identify any major differences between the row 'Demand w/o Buffer' under the 'Measures' row for the material number across the different weeks in the first column.
            3.  Distinguish between the effects of demand variability and supply-side delays or gaps.
            4.  If there are any negative values for the 'EOH w/o Buffer', validate within the same week if there are any supply issues or negative inventory or demand requirements.
            5.  Provide contextual interpretations—do not just state values.
            6.  Provide recommendations focusing on improving the overall supply to meet demand fluctuations.
            7.  Infer root causes for observed issues, particularly:
            * Why does the inventory on hand - 'EOH w/o Buffer' - turn negative and remains negative (e.g., zero supply, poor planning, or missed lead times, unexpected demand)
            
            Integrate into your analysis the understanding that while the waterfall chart visually represents the cumulative effect of sequential changes, the actual root causes should be inferred from the underlying data and context. Do not simply state this explanation as a separate point; weave it into your analysis.
            Do not give me generic explanations. Everything has to be backed with the data you have seen. No generic ifs and hows.
            
            Important: Use the sequential weekly data to derive temporal insights. Understand that the waterfall-style cumulative effect seen in EOH charts reflects decisions made in earlier weeks. Your analysis must go beyond visualization and into operational logic.

            Do not include any introductory phrases or preambles. Start directly with the bullet points.
            """

    user_prompt = f"""
        Explain the root cause analysis for the following waterfall chart data:\n\n{df_string}
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
        st.header("Root Cause Analysis")
        st.write(explanation)

    except Exception as e:
        st.error(f"Error during Groq API call: {e}")

def explain_inventory_events(representative_weekly_events, reorder_point, lead_time, lead_time_std_dev, consumption_distribution_params, consumption_type, consumption_best_distribution, order_distribution_params, order_quantity_type, order_distribution_best):
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
            * Quantify the difference between forecasted and actual consumption. Actual consumption is the simulated consumption based on the fixed or distribution identified and forecasted consumption is using the forecasting models. 
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
            * Explain how the provided forecasted consumption data could have been better used.
            * ONLY use the data provided in the weekly events to support your analysis. Do not introduce any external information or assumptions.
        * Inventory Policy Recommendations:
            * Provide specific, data-driven recommendations for inventory policies to prevent future stockouts for the given sequence of events.
            * Suggest optimal reorder points, safety stock levels, and order quantities for both reactive and proactive inventory for the given data or how the order placement should have been done better.
            * Recommend adjustments to the forecasting method based on the observed consumption patterns.
            * Recommend a policy that accounts for the variation in the consumption.
            * Recommend a value in which initial inventory should have started off to prevent stockouts. Reason the math mildly.
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

    def process_chunk(chunk):
        weekly_events_text = "\n\n".join(chunk)
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
            return chat_completion.choices[0].message.content
        except Exception as e:
            return f"Error during Groq API call: {e}"

    try:
        # Attempt to process the entire list of events
        result = process_chunk(representative_weekly_events)
        st.write(result)
    except Exception as e:
        # If an exception occurs, split the events into chunks and process each chunk separately
        max_divisor = 10  # Maximum divisor to split the data
        for divisor in range(2, max_divisor + 1):
            chunk_size = len(representative_weekly_events) // divisor
            chunks = [representative_weekly_events[i:i + chunk_size] for i in range(0, len(representative_weekly_events), chunk_size)]

            results = []
            for chunk in chunks:
                result = process_chunk(chunk)
                if "Error during Groq API call" not in result:
                    results.append(result)
                else:
                    break  # If any chunk fails, break and try a smaller chunk size

            if len(results) == len(chunks):  # If all chunks processed successfully
                break  # Exit the loop

        # Consolidate the results into a single cohesive report
        consolidated_analysis = {
            "Stockout/Near-Stockout Events": [],
            "Consumption Patterns": [],
            "Inventory Levels": [],
            "Proactive vs. Reactive Inventory": [],
            "Inventory Policy Recommendations": [],
            "Ordering Strategy Improvements": [],
            "Summary of Key Insights and Recommendations": []
        }

        for result in results:
            sections = result.split("\n\n")
            for section in sections:
                if "Stockout/Near-Stockout Events" in section:
                    consolidated_analysis["Stockout/Near-Stockout Events"].append(section)
                elif "Consumption Patterns" in section:
                    consolidated_analysis["Consumption Patterns"].append(section)
                elif "Inventory Levels" in section:
                    consolidated_analysis["Inventory Levels"].append(section)
                elif "Proactive vs. Reactive Inventory" in section:
                    consolidated_analysis["Proactive vs. Reactive Inventory"].append(section)
                elif "Inventory Policy Recommendations" in section:
                    consolidated_analysis["Inventory Policy Recommendations"].append(section)
                elif "Ordering Strategy Improvements" in section:
                    consolidated_analysis["Ordering Strategy Improvements"].append(section)
                elif "Summary of Key Insights and Recommendations" in section:
                    consolidated_analysis["Summary of Key Insights and Recommendations"].append(section)

        # Merge the sections into a single cohesive report
        final_report = ""
        for section, content in consolidated_analysis.items():
            final_report += f"{section}:\n"
            for item in content:
                final_report += f"{item}\n\n"

        st.write(final_report)