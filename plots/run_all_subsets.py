#run command line

import subprocess


subsets=["AMO18", "GRO20", "GRO21", "RYB20", "RYB21"]
for subset in subsets:
    subprocess.run("jupyter nbconvert plots/process.ipynb --to python")
    subprocess.run(f"ipython plots/process.py {subset}")