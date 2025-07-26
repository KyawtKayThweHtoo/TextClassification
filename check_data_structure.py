import pandas as pd
import os

def check_file_structure(filepath):
    try:
        # Read just the first row to get column names
        df = pd.read_excel(filepath, nrows=1)
        return {
            'file': os.path.basename(filepath),
            'columns': df.columns.tolist(),
            'row_count': sum(1 for _ in pd.read_excel(filepath, chunksize=1000))
        }
    except Exception as e:
        return {'file': os.path.basename(filepath), 'error': str(e)}

# Check all Excel files in the data directory
data_dir = 'data'
for filename in os.listdir(data_dir):
    if filename.endswith('.xlsx'):
        filepath = os.path.join(data_dir, filename)
        result = check_file_structure(filepath)
        print(f"\nFile: {result['file']}")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Columns: {result['columns']}")
            print(f"Number of rows: {result['row_count']}")
