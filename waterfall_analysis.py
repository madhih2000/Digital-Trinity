import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from io import BytesIO
from openpyxl import load_workbook
from openpyxl.styles import PatternFill

def generate_weeks_range(start_week, num_weeks=12):
    #weeks_range = [f"WW{str((start_week + i - num_weeks) % 52 or 52).zfill(2)}" for i in range(2 * num_weeks + 1)]
    weeks_range = []
    for i in range(2 * num_weeks + 1):
        week = (start_week + i - num_weeks) % 52 or 52
        if start_week-num_weeks < 1 and week > start_week + num_weeks: #edge case in case you want to check shortage data for week 2 ,e.g.
            continue
        weeks_range.append(f"WW{str(week).zfill(2)}")
    print(weeks_range)
    return weeks_range

# Function to normalize column names
def normalize(col):
    return col.strip().replace('\n', ' ').replace('\r', ' ').replace('\t', ' ').lower()

def save_to_excel(df, output_file):
    """Saves a DataFrame to an Excel file."""
    if df is not None:
        try:
            df.to_excel(output_file, index=False)
            print(f"Data saved to '{output_file}'.")
        except Exception as e:
            print(f"Error saving to Excel: {e}")
    else:
        print("No data to save.")

def extract_and_aggregate_weekly_data(folder_path, material_number, plant, site, start_week, num_weeks=12):
    """
    Extracts and aggregates weekly data for a specific material number, plant, and site,
    starting from a specified week and including the next 'num_weeks' weeks.

    Args:
        folder_path (str): Path to the folder containing the XLSX files.
        material_number (str): Material number to filter.
        plant (str): Plant to filter.
        site (str): Site to filter.
        start_week (str): Starting week (e.g., "WW32").
        num_weeks (int): Number of weeks to include in the output.

    Returns:
        pandas.DataFrame: DataFrame containing the aggregated weekly data, or None if no data is found.
    """

    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return None
    
    print(folder_path)

    all_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".xlsx")])

    if not all_files:
        print(f"Error: No XLSX files found in '{folder_path}'.")
        return None

    week_numbers = list(range(max(1, start_week - num_weeks), start_week + 1))
    #print(week_numbers)

    selected_data = []
    selected_weeks = []

    #Columns to add to the final df
    initial_columns = ["MaterialNumber", "Plant", "Site", "Measures", "InventoryOn-Hand"]
    weeks_range = generate_weeks_range(start_week,num_weeks)
    #print(weeks_range)

    for i in week_numbers:
        week_file = f"WW{i}.xlsx"
        week_col = f"WW{i:02d}"

        if week_file in all_files:
            file_path = os.path.join(folder_path, week_file)
            try:
                df = pd.read_excel(file_path)
                filtered_df = df[
                    (df["Material Number"] == material_number)
                    & (df["Plant"] == plant)
                    & (df["Site"] == site)
                ]

                lead_value = 0

                if i == start_week:

                    # Lead Time column, normalized
                    target_col = normalize("Lead Time (Week)")
                    lead_col = None

                    # Loop to find the unnormalized matching column
                    for col in df.columns:
                        if normalize(col) == target_col:
                            lead_col = col
                            break

                    if lead_col:
                        lead_value = df.loc[df["Material Number"] == material_number, lead_col]

                
                #Remove whitespace in column names
                filtered_df.columns = filtered_df.columns.str.strip().str.replace('\n', ' ', regex=False).str.replace(r'\s+', '', regex=True)
        
                if not filtered_df.empty:
                    # Select only the required columns
                    all_columns = initial_columns + weeks_range

                    # Check if all columns exist in filtered_df
                    missing_columns = [col for col in all_columns if col not in filtered_df.columns]
                    
                    if missing_columns:
                        raise ValueError(f"Missing columns in {week_file}: {', '.join(missing_columns)}")
                        return None
                    
                    filtered_df = filtered_df[all_columns]

                    #Add new column snapshot
                    filtered_df['Snapshot'] = week_col

                    # Remove the first element from weeks_range for next iteration
                    weeks_range = weeks_range[1:]

                    selected_data.append(filtered_df)
                    selected_weeks.append(f"WW{i:02d}")

                else:
                    # If filtered_df is empty, create a new dataframe with NaN values
                    all_columns = initial_columns + weeks_range
                    empty_df = pd.DataFrame(columns=all_columns)
                    empty_df['Snapshot'] = week_col  # Add the Snapshot column with the week_col value

                    # Add the empty dataframe to selected_data and weeks
                    selected_data.append(empty_df)
                    selected_weeks.append(f"WW{i:02d}")

            except FileNotFoundError:
                print(f"Error: File '{week_file}' not found.")
            except Exception as e:
                print(f"Error reading file '{week_file}': {e}")
        else:
            print(f"Warning: Week file '{week_file}' not found.")

    if not selected_data:
        print("No matching data found.")
        return None

    result_df = pd.concat(selected_data, ignore_index=True)
    
    #reorder columns to have week at the beginning.
    cols = result_df.columns.tolist()
    cols = ['Snapshot'] + [col for col in cols if col != 'Snapshot']
    result_df = result_df[cols]

    return result_df, lead_value

def plot_stock_prediction_plotly(df, start_week, lead_time, weeks_range):
    """Plots actual and predicted stock values over weeks."""

    weeks = generate_weeks_range(start_week, weeks_range)
    stock_values = df[df['Measures'] == 'Weeks of Stock']

    fig = go.Figure()

    actual_values = []
    predicted_values = []
    plot_weeks_actual = []
    plot_weeks_predicted = []

    for i, week in enumerate(weeks):
        print(i)
        print(predicted_values)
        if week not in df.columns:
            print(f"Warning: Week {week} not found in DataFrame.")
            continue  # skip the iteration if the week is not in the dataframe.

        if week in df['Snapshot'].values: # Only get actual values if the week is in the snapshot.
            actual_value = stock_values[stock_values['Snapshot'] == week][week].iloc[0]
            if actual_value is None or np.isnan(actual_value):
                actual_value = 0  # if actual value is none or nan, set to 0.
            actual_values.append(actual_value)
            plot_weeks_actual.append(week)
        else:
            break

        if i != 0:
            predicted_value = stock_values[stock_values['Snapshot'] == weeks[i - 1]][week].iloc[0]
        else:
            predicted_value = None

        if predicted_value is None or np.isnan(predicted_value):
            if i > 0:
                predicted_value = predicted_values[-1]  # use the previous predicted value if the current one is None or NaN
            else:
                predicted_value = 0  # skip the first one as it is none.

        predicted_values.append(predicted_value)
        plot_weeks_predicted.append(week)

    fig.add_trace(go.Scatter(x=plot_weeks_actual, y=actual_values, mode='lines+markers', name='Actual Weeks of Stock'))
    fig.add_trace(go.Scatter(x=plot_weeks_predicted, y=predicted_values, mode='lines+markers', name='Predicted Weeks of Stock'))

    fig.update_layout(title='Actual vs. Predicted Weeks of Stock',
                      xaxis_title='Week',
                      yaxis_title='Weeks of Stock')

    return actual_values, fig

def check_wos_against_lead_time(wos_list, lead_time):
    """
    Compares values in wos_list against lead_time and flags close matches.

    Args:
        wos_list: A list of floats, potentially containing 0.
        lead_time: An integer representing the target lead time.

    Returns:
        A list of messages indicating the comparison results and a boolean indicating
        whether an immediate order is needed.
    """
    # Check if lead_time is a pandas Series and handle it
    if isinstance(lead_time, pd.Series):
        lead_time_value = lead_time.iloc[0]
    else:
        lead_time_value = lead_time  # If it's already a scalar


    messages = []
    order_immediately = False
    tolerance = 0.1  # 10% tolerance

    for i, wos in enumerate(wos_list):
        if wos == 0:
            #messages.append(f"Index {i}: No value recorded.")
            continue

        lower_bound = lead_time_value * (1 - tolerance)
        upper_bound = lead_time_value * (1 + tolerance)

        if wos < lower_bound:
            messages.append(f"Index {i}: Value {wos} is significantly lower than lead time {lead_time_value}. ALERT!")
        elif wos > upper_bound:
            continue
            #messages.append(f"Index {i}: Value {wos} is slightly higher than lead time {lead_time_value}. Warning.")

    if wos_list and wos_list[-1] != 0:
        if wos_list[-1] < lead_time_value*0.5: # if the last value is less than half the lead time.
            order_immediately = True
            messages.append("Last recorded value is significantly low, consider immediate order.")
        else:
            messages.append("Last recorded value does not indicate immediate order is needed.")

    return messages, order_immediately

def apply_coloring_to_output(excel_buffer, lead_time):
    # Rewind buffer and load workbook
    excel_buffer.seek(0)
    wb = load_workbook(excel_buffer)
    ws = wb['Sheet1']

    # Define colors
    red_fill = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
    yellow_fill = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    green_fill = PatternFill(start_color="00FF00", end_color="00FF00", fill_type="solid")

    # Identify relevant columns
    header = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
    measures_col_idx = header.index('Measures') + 1
    ww_col_indices = [i + 1 for i, h in enumerate(header) if str(h).startswith("WW")]

    # Apply coloring logic
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        if row[measures_col_idx - 1].value == 'Weeks of Stock':
            for idx in ww_col_indices:
                cell = row[idx - 1]
                raw_val = cell.value
                try:
                    val = float(raw_val)
                except (TypeError, ValueError):
                    continue  # Skip non-numeric

                
                if val < 0:
                    cell.fill = red_fill
                if val < lead_time.iloc[0]:
                    cell.fill = yellow_fill
                else:
                    cell.fill = green_fill

    # Save again to a new BytesIO
    colored_output = BytesIO()
    wb.save(colored_output)
    colored_output.seek(0)
    return colored_output