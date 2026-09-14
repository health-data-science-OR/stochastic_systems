'''
Simple simulation output analysis code 

This includes:

`create_user_controlled_hist` to create a plotly
histogram of all of the replication results.
'''

import pandas as pd
import plotly.graph_objects as go

def create_user_controlled_hist(
    results, exclude_columns=None, name_mappings=None, include_instruct=True
):
    """
    Create a plotly histogram that includes a drop down list that allows a user
    to select which KPI is displayed as a histogram

    Params:
    -------
    results: pd.DataFrame
        rows = replications, cols = KPIs
    exclude_columns: list, optional
        List of column numbers to exclude from the dropdown list
    name_mappings: dict, optional
        Nested dictionary mapping column names to friendly names and units
        Format: {column_name: {'friendly_name': str, 'units': str}}#
    include_instruct: bool, optional
        Including the instruction "Select KPI from drop down list" above
        plot. Useful for interactive applications.

    Returns:
    -------
    plotly.figure

    Source:
    ------
    The code in this function was adapted from:
    https://stackoverflow.com/questions/59406167/plotly-how-to-filter-a-pandas-
    dataframe-using-a-dropdown-menu

    and

    Monks T and Harper A. Improving the usability of open health service 
    delivery simulation models using Python and web apps 
    [version 2; peer review: 3 approved]. NIHR Open Res 2023, 3:48 #
    https://doi.org/10.3310/nihropenres.13467.2
    """

    # create a figure
    fig = go.Figure()

    # Filter out excluded columns
    if exclude_columns is None:
        exclude_columns = []
    included_columns = [
        col
        for i, col in enumerate(results.columns)
        if i not in exclude_columns
    ]

    # set up ONE trace
    first_col = included_columns[0]
    first_friendly_name = (
        name_mappings[first_col]["friendly_name"]
        if name_mappings and first_col in name_mappings
        else first_col
    )
    first_units = (
        name_mappings[first_col]["units"]
        if name_mappings and first_col in name_mappings
        else ""
    )
    first_x_title = (
        f"{first_friendly_name} ({first_units})"
        if first_units
        else first_friendly_name
    )

    fig.add_trace(go.Histogram(x=results[first_col]))
    fig.update_xaxes(title_text=first_x_title)

    buttons = []

    # create list of drop down items - KPIs
    for col in included_columns:
        if name_mappings and col in name_mappings:
            friendly_name = name_mappings[col]["friendly_name"]
            units = name_mappings[col]["units"]
            x_title = f"{friendly_name} ({units})" if units else friendly_name
        else:
            friendly_name = col
            x_title = col

        buttons.append(
            dict(
                method="update",
                label=friendly_name,
                args=[{"x": [results[col]]}, {"xaxis": {"title": x_title}}],
            )
        )

    # create update menu and parameters
    updatemenu = []
    your_menu = dict()
    updatemenu.append(your_menu)

    updatemenu[0]["buttons"] = buttons
    updatemenu[0]["direction"] = "down"
    updatemenu[0]["showactive"] = True

    # add dropdown menus to the figure
    fig.update_layout(showlegend=False, updatemenus=updatemenu)

    # Add annotation as instruction
    if include_instruct:
        fig.add_annotation(
            text="Select a KPI from the drop down list",
            xref="paper",
            yref="paper",
            x=0.0,
            y=1.1,  # Positions the text above the plot
            showarrow=False,
            font=dict(size=12),
        )

    return fig


def experiment_summary_frame(experiment_results):
    """
    Mean results for each performance measure by experiment

    Parameters:
    ----------
    experiment_results: dict
        dictionary of replications.
        Key identifies the performance measure

    Returns:
    -------
    pd.DataFrame
    """
    columns = []
    summary = pd.DataFrame()
    for sc_name, replications in experiment_results.items():
        summary = pd.concat([summary, replications.mean()], axis=1)
        columns.append(sc_name)

    summary.columns = columns
    return summary

def create_example_csv(filename="example_experiments.csv"):
    """
    Create an example CSV file to use in tutorial.
    This creates 4 experiments that varys
    n_operators, and mean_iat.

    Params:
    ------
    filename: str, optional (default='example_experiments.csv')
        The name and path to the CSV file.
    """
    # each column is defined as a seperate list
    names = ["base", "op+1", "high_demand", "combination"]
    operators = [13, 14, 13, 14]
    nurses = [9, 9, 9, 9]
    mean_iat = [0.6, 0.6, 0.55, 0.55]
    chance_callback = [0.4, 0.4, 0.4, 0.4]

    # empty dataframe
    df_experiments = pd.DataFrame()

    # create new columns from lists
    df_experiments["experiment"] = names
    df_experiments["n_operators"] = operators
    df_experiments["n_nurses"] = nurses
    df_experiments["mean_iat"] = mean_iat
    df_experiments["chance_callback"] = chance_callback

    return df_experiments





