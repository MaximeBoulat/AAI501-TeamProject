import numpy as np

# Convert NumPy types in results to native Python types
def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)

json.dump(results, f, indent=2, default=convert)

