import sys
import pandas as pd

# # Add the path to the `scripts.py` module to the system path
sys.path.append(r"C:\Users\modri\Desktop\python\Peregrin\Peregrin-1\peregrin\peregrin")


# Import the required classes from `scripts.py`
from scripts import PlotParams

# Create a DataFrame
data = {'column': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 'column2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
df = pd.DataFrame(data)

try:
    # Create an instance of PlotParams and calculate x_span
    x_span_value = PlotParams.x_span(df)
    print(f"Calculated x_span: {x_span_value}")

except FileNotFoundError:
    print("The input file was not found. Please check the path.")
except Exception as e:
    print(f"An error occurred: {e}")