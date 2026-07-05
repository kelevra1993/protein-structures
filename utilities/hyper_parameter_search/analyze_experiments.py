import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path

# ==========================================
# CONFIGURATION
# ==========================================
# Set the target folder containing the experiment runs
EXPERIMENTS_FOLDER = "/home/robert_kelevra/Projects/protein-structures/data_examples/protein_experiments"

# Output HTML file name
OUTPUT_HTML = "experiment_results.html"

# Smoothing parameter for the curves (Exponential Moving Average)
# 0.0 means no smoothing, 0.9 means very heavy smoothing
SMOOTHING_ALPHA = 0.8

# The hyperparameter to color code the curves by (must match a key in the JSON).
# Set to None if you just want completely unique colors for every line.
COLOR_BY_HYPERPARAMETER = "learning_rate" 

def smooth_curve(scalars, weight):
    """
    Exponential Moving Average (EMA) implementation to smooth noisy metrics.
    weight is between 0 and 1. Higher weight = more smoothing.
    """
    if weight == 0:
        return scalars
    if not scalars:
        return scalars
        
    last = scalars[0]
    smoothed = []
    for point in scalars:
        # Avoid NaN issues
        if pd.isna(point):
            smoothed.append(point)
            continue
            
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def analyze_experiments():
    base_dir = Path(EXPERIMENTS_FOLDER)
    
    if not base_dir.exists() or not base_dir.is_dir():
        print(f"Error: The folder '{EXPERIMENTS_FOLDER}' does not exist in the current directory.")
        print(f"Please update the EXPERIMENTS_FOLDER variable in the script or run this from the correct path.")
        return

    summary_data = []
    timeseries_data = []

    print(f"Scanning directory: {base_dir}")
    
    # Iterate through all subdirectories
    for run_dir in base_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        json_path = run_dir / "hyper_parameter_configuration.json"
        csv_path = run_dir / "metrics_evolution.csv"
        
        # Dodge the folder if missing required files
        if not json_path.exists() or not csv_path.exists():
            continue
            
        # 1. Load Hyperparameters
        with open(json_path, "r") as f:
            try:
                hyperparams = json.load(f)
            except json.JSONDecodeError:
                print(f"Failed to parse JSON in {run_dir.name}. Skipping.")
                continue
            
        # 2. Load Metrics
        try:
            df_metrics = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Failed to read CSV for {run_dir.name}: {e}")
            continue
            
        if "Mean_Angle_Delta" not in df_metrics.columns or "Mean_Distance_Delta" not in df_metrics.columns:
            print(f"Warning: Missing required columns in {run_dir.name}. Skipping.")
            continue
            
        # Calculate summary statistics (Mean and Min as requested)
        avg_angle = round(df_metrics["Mean_Angle_Delta"].mean(),4)
        min_angle = df_metrics["Mean_Angle_Delta"].min()
        
        avg_distance = round(df_metrics["Mean_Distance_Delta"].mean(),4)
        min_distance = df_metrics["Mean_Distance_Delta"].min()
        
        # Compile row for the DataFrame
        row = hyperparams.copy()
        row["run_name"] = run_dir.name
        row["avg_angle_delta"] = avg_angle
        row["min_angle_delta"] = min_angle
        row["avg_distance_delta"] = avg_distance
        row["min_distance_delta"] = min_distance
        
        summary_data.append(row)
        
        # Store full timeseries for plotting
        timeseries_data.append({
            "run_name": run_dir.name,
            "hyperparams": hyperparams,
            "angle_curve": df_metrics["Mean_Angle_Delta"].tolist(),
            "distance_curve": df_metrics["Mean_Distance_Delta"].tolist(),
            "iterations": df_metrics.index.tolist() # X-axis
        })

    if not summary_data:
        print("No valid experiment folders found with both JSON and CSV files.")
        return

    # Create Summary DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Reorder columns to put metrics at the front for easier reading
    metric_cols = ["run_name", "avg_angle_delta", "min_angle_delta", "avg_distance_delta", "min_distance_delta"]
    param_cols = [c for c in df_summary.columns if c not in metric_cols]
    df_summary = df_summary[metric_cols + param_cols]
    
    print("\n--- Summary DataFrame ---")
    print(df_summary.to_string(index=False))
    
    # Export DataFrame to CSV for convenience
    csv_out_path = "experiment_summary.csv"
    df_summary.to_csv(csv_out_path, index=False)
    print(f"\nSummary data exported to: {csv_out_path}")
    
    # ==========================================
    # INTERACTIVE PLOTTING (Plotly)
    # ==========================================
    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Mean Angle Delta Evolution", "Mean Distance Delta Evolution")
    )
    
    colors_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24

    # 1. Pre-add all traces without specific colors/names
    for run in timeseries_data:
        hp_str = "<br>".join([f"{k}: {v}" for k, v in run["hyperparams"].items()])
        hover_template = (
            f"<b>{run['run_name']}</b><br><br>"
            f"<u>Hyperparameters:</u><br>{hp_str}<br><br>"
            f"Iteration: %{{x}}<br>"
            f"Value: %{{y:.4f}}<extra></extra>"
        )
        angle_vals = smooth_curve(run["angle_curve"], SMOOTHING_ALPHA)
        dist_vals = smooth_curve(run["distance_curve"], SMOOTHING_ALPHA)
        
        # Plot 1: Angle Curve
        fig.add_trace(go.Scatter(x=run["iterations"], y=angle_vals, mode="lines", hovertemplate=hover_template), row=1, col=1)
        # Plot 2: Distance Curve
        fig.add_trace(go.Scatter(x=run["iterations"], y=dist_vals, mode="lines", hovertemplate=hover_template), row=1, col=2)

    # 2. Build the dropdown buttons for every hyperparameter
    buttons = []
    all_coloring_options = ["run_name"] + param_cols
    
    for param in all_coloring_options:
        unique_vals = df_summary[param].dropna().unique()
        try:
            unique_vals = sorted(unique_vals)
        except TypeError:
            pass # Mixed types, skip sorting
            
        color_map = {str(val): colors_palette[j % len(colors_palette)] for j, val in enumerate(unique_vals)}
        
        line_colors = []
        names = []
        legendgroups = []
        showlegends = []
        seen_vals = set()
        
        for run in timeseries_data:
            val = str(run["run_name"]) if param == "run_name" else str(run["hyperparams"].get(param, "Unknown"))
            c = color_map.get(val, "#888888")
            label = f"{val}"
            
            # Angle Trace
            line_colors.append(c)
            names.append(label)
            legendgroups.append(label)
            if val not in seen_vals:
                showlegends.append(True)
                seen_vals.add(val)
            else:
                showlegends.append(False)
                
            # Distance Trace
            line_colors.append(c)
            names.append(label)
            legendgroups.append(label)
            showlegends.append(False)
            
        # Plotly restyle method assigns a list of attributes to the list of traces
        buttons.append(dict(
            label=param,
            method="restyle",
            args=[
                {
                    "line.color": line_colors,
                    "name": names,
                    "legendgroup": legendgroups,
                    "showlegend": showlegends
                }
            ]
        ))
        
    # 3. Apply the first parameter ("run_name") as the default view
    if buttons:
        default_args = buttons[0]["args"][0]
        for i in range(len(fig.data)):
            fig.data[i].line.color = default_args["line.color"][i]
            fig.data[i].name = default_args["name"][i]
            fig.data[i].legendgroup = default_args["legendgroup"][i]
            fig.data[i].showlegend = default_args["showlegend"][i]

    # 4. Layout Styling
    fig.update_layout(
        title=f"Experiment Curves | Smoothing: {SMOOTHING_ALPHA}",
        hovermode="closest",
        template="plotly_dark",
        updatemenus=[
            dict(
                active=0,
                buttons=buttons,
                x=1.02,
                y=1.15,
                xanchor="left",
                yanchor="top",
                direction="down",
                showactive=True,
                bgcolor="#333333",
                bordercolor="#555555",
                font=dict(color="#ffffff")
            )
        ],
        annotations=[
            dict(
                text="<b>Color By:</b>",
                x=1.02,
                y=1.18,
                xref="paper",
                yref="paper",
                showarrow=False,
                xanchor="left"
            )
        ],
        margin=dict(r=150) # Add some right margin so the dropdown isn't cut off
    )
    
    fig.update_xaxes(title_text="Iteration", row=1, col=1)
    fig.update_xaxes(title_text="Iteration", row=1, col=2)
    fig.update_yaxes(title_text="Angle Delta", row=1, col=1)
    fig.update_yaxes(title_text="Distance Delta", row=1, col=2)

    # ==========================================
    # PARALLEL COORDINATES PLOT
    # ==========================================
    pcp_dimensions = []
    
    # Process hyperparameter dimensions
    for col in param_cols:
        series = df_summary[col].dropna()
        if series.empty:
            continue
            
        if pd.api.types.is_numeric_dtype(series) and not set(series.unique()).issubset({True, False}):
            # Numeric parameter
            pcp_dimensions.append(dict(label=col, values=df_summary[col]))
        else:
            # Categorical or boolean parameter, map to integers for plotting
            unique_vals = list(series.unique())
            try:
                unique_vals = sorted(unique_vals)
            except TypeError:
                pass
            val_to_int = {val: i for i, val in enumerate(unique_vals)}
            num_values = df_summary[col].map(val_to_int)
            pcp_dimensions.append(dict(
                label=col,
                tickvals=list(range(len(unique_vals))),
                ticktext=[str(v) for v in unique_vals],
                values=num_values
            ))
            
    # Add our primary metric at the end so it's the right-most axis
    pcp_dimensions.append(dict(
        label="Min Distance Delta",
        values=df_summary["min_distance_delta"]
    ))
    
    fig_pcp = go.Figure(data=
        go.Parcoords(
            line=dict(
                color=df_summary["min_distance_delta"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Min Distance")
            ),
            dimensions=pcp_dimensions
        )
    )
    
    fig_pcp.update_layout(
        title="Hyperparameter Parallel Coordinates Plot (Color: Min Distance Delta)",
        template="plotly_dark",
        margin=dict(t=60, b=40)
    )

    # ==========================================
    # SAVE TO HTML DASHBOARD
    # ==========================================
    out_path = Path(OUTPUT_HTML)
    
    # We combine both figures into a single HTML file by extracting their HTML snippets
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("<html><head><title>Experiment Analysis Dashboard</title></head>")
        f.write("<body style='background-color: #111111; color: white; font-family: sans-serif; margin: 0; padding: 20px;'>")
        f.write("<h1 style='text-align: center; margin-bottom: 30px;'>Hyperparameter Tuning Dashboard</h1>")
        
        # 1. Plotly Training Curves
        f.write(fig.to_html(full_html=False, include_plotlyjs='cdn'))
        
        f.write("<hr style='border-color: #333333; margin: 40px 0;'>")
        
        # 2. Parallel Coordinates Plot
        f.write("<div style='margin-bottom: 20px;'><p style='text-align: center; color: #aaaaaa;'>")
        f.write("<i>Drag the axes to reorder them, or click and drag along the axes to filter specific ranges!</i>")
        f.write("</p></div>")
        f.write(fig_pcp.to_html(full_html=False, include_plotlyjs=False)) # Plotly JS is already included
        
        f.write("</body></html>")
        
    print(f"Interactive dashboard successfully saved to: {out_path.absolute()}")
    print("Open this HTML file in any web browser to view your interactive charts.")

if __name__ == "__main__":
    analyze_experiments()
