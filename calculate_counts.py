import pandas as pd
import os

# --- Analysis of individual Excel files in 'data' directory ---
def analyze_excel_files():
    print("--- Analyzing Individual Excel Files in 'data' Directory ---")
    data_dir = 'data'
    grand_total_papers = 0
    papers_per_file = {}

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    # Find all .xlsx files in the data directory
    excel_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.xlsx')]
    if not excel_files:
        print("No Excel (.xlsx) files found in the data directory.")
        return

    for file_name in excel_files:
        file_path = os.path.join(data_dir, file_name)
        try:
            df = pd.read_excel(file_path)
            count = len(df)
            papers_per_file[file_name] = count
            grand_total_papers += count
            print(f"Found {count} papers in '{file_name}'.")
        except Exception as e:
            print(f"Error reading '{file_name}': {e}")
            papers_per_file[file_name] = 0

    print(f"\nGrand total papers from ALL Excel files in '{data_dir}': {grand_total_papers}")
    print("---------------------------------------------------------")

# --- Analysis of preprocessed_papers_data.csv (for comparison) ---
def analyze_preprocessed_csv():
    print("\n--- Analyzing 'preprocessed_papers_data.csv' (as used by app.py) ---")
    # Thesis fields as defined in app.py
    FIELDS = [
        'Artificial Intelligence',
        'Distribution Data',
        'Image Preprocessing',
        'Networking and Cybersecurity',
        'Software Engineering'
    ]
    try:
        data_df = pd.read_csv('preprocessed_papers_data.csv')
        initial_total_papers = len(data_df)
        print(f"Actual total rows in preprocessed_papers_data.csv (before any filtering): {initial_total_papers}")

        data_df_filtered = data_df.dropna(subset=['Title', 'Abstract', 'Category'])
        total_papers_after_filtering = len(data_df_filtered)
        print(f"Total papers from preprocessed_papers_data.csv after filtering: {total_papers_after_filtering}")

        category_counts = data_df_filtered['Category'].value_counts()

        print("\nPapers per specified field (from preprocessed_papers_data.csv after filtering):")
        for field in FIELDS:
            count = category_counts.get(field, 0)
            print(f"- {field}: {count}")

        print("\nAll categories found in preprocessed_papers_data.csv and their counts (after filtering):")
        if not category_counts.empty:
            for category, count_val in category_counts.items():
                print(f"- {category}: {count_val}")
        else:
            print("No categories found or the 'Category' column is empty after filtering.")

    except FileNotFoundError:
        print("Error: 'preprocessed_papers_data.csv' not found.")
    except Exception as e:
        print(f"An error occurred while processing 'preprocessed_papers_data.csv': {e}")
    print("-------------------------------------------------------------------")

if __name__ == '__main__':
    analyze_excel_files()
    analyze_preprocessed_csv()
