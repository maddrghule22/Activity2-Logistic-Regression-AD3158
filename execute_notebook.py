"""
Executes Activity2_Logistic_Regression.ipynb and saves output cells.
"""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

print("Loading notebook...")
with open("Activity2_Logistic_Regression.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)

ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
print("Executing all notebook cells...")
ep.preprocess(nb, {'metadata': {'path': '.'}})

with open("Activity2_Logistic_Regression.ipynb", "w", encoding="utf-8") as f:
    nbformat.write(nb, f)

print("✅ Notebook executed successfully and all outputs recorded!")
