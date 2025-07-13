import json
import os
import pandas as pd

def save_results_to_json(results, operation_type, filename="operations.json"):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            data = json.load(f)
    else:
        data = {"results": []}
    
    result_entry = {
        "operation": operation_type,
        "timestamp": pd.Timestamp.now().isoformat(),
        "data": results
    }
    
    data["results"].append(result_entry)
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    return filename