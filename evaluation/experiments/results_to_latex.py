import os
import numpy as np
import pandas as pd
import argparse

# Pretty names for metrics and models
metric_pretty_names = {
    "spectral": "spectral",
    "degree": "degree",
    "degree_centrality": "deg. cent.",
    "clustering": "local clust. coeff.",
    "closeness_centrality": "close. cent.",
    "katz_centrality": "katz cent.",
    "eigenvector_centrality": "eigen. cent.",
    "avg_clust_coeff": "ave. clust. coeff.",
    "transitivity": "transitivty",
    "diameter": "diameter",
    "average_shortest_path_length": "ave. short. path length",
    "link_forecasting": "link pred. AUC",
    "temporal_correlation": "temp. corr.",
    "diff_avg_temporal_closeness": "temp. close. diff.",
    "diff_avg_temporal_clustering_coefficient": "temp. clust. coeff. diff.",
    "NNDR_mean": "NNDR (mean ± std)",
    "NNDR_std": "NNDR (mean ± std)",
}

model_pretty_names = {
    "damnets": "DAMNETS",
    "damcc": "DAMCC",
    "dymond": "DYMOND",
    "d2g2": "D2G2",
    "tigger": "TIGGER",
    "age": "AGE",
    "random": "RANDOM"
}

# Boolean to check if the model is an update model
is_update_model = {
    "damnets": True,
    "age": True,
    "damcc": True,
    "random": True, 
    "dymond": False,
    "d2g2": False,
    "tigger": False,
}

def main(dataset_name: str) -> None:
    full_metrics = pd.DataFrame()

    # Load all the CSV files and concatenate their results
    for file in os.listdir(os.path.join("generated_graphs", "metrics")):
        series = pd.read_csv(os.path.join("generated_graphs", "metrics", file), index_col=0)
        
        # Extract the model and dataset name from the filename
        column = series.columns[0]
        if '-' in column:
            model, dataset = column.split('-')
            note = ""
        else:
            name_, note = column.split("_") if "_" in column else (column, "")
            model, dataset = name_.split("-")

        # Skip if the dataset doesn't match the one we're interested in
        if dataset.strip() != dataset_name.strip():
            continue

        # Store the series in the DataFrame with proper model name
        full_metrics[f"{model}{f'_{note}' if note != '' else ''}"] = series

    # Drop rows where all values are NaN
    full_metrics = full_metrics.dropna(axis=0)

    # Reorder the columns as per the desired order
    desired_order = ["age", "damcc", "damnets", "random"]
    full_metrics = full_metrics[desired_order]

    # LaTeX table header
    print("\\toprule")
    print(" & \\multicolumn{4}{c}{\\textit{Full series}} \\\\")
    print("\\cmidrule{2-5}")
    print(" & AGE & DAMCC & DAMNETS & RANDOM \\\\")
    print("\\midrule")

    # Function to format numbers for LaTeX
    def number_format(x):
        return f"{x:.3f}" if not pd.isna(x) else "-"

    # Generate the LaTeX table rows
    for label, row in full_metrics.iterrows():
        higher_is_better = label in ["link_forecasting", "NNDR_std", "NNDR_mean"]
        arrow = '$\\nearrow$' if higher_is_better else '$\\searrow$'
        row_values = " & ".join(
            [f"\\textbf{{{number_format(val)}}}" if val == (row.min() if not higher_is_better else row.max()) else number_format(val) for val in row.values]
        )
        print(f"{arrow} \\textit{{{metric_pretty_names[label]}}} & {row_values}\\\\")
    
    print("\\bottomrule")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ba")
    args = parser.parse_args()
    
    main(args.dataset)
