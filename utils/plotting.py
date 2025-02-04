import numpy as np

def evaluate_function(function_str):
    """Evaluate the user-supplied mathematical function."""
    try:
        x = np.linspace(-100, 100, 500)
        y = eval(function_str)  # Caution: Only safe if function input is sanitized
        return {"x": x.tolist(), "y": y.tolist(), "error": None}
    except Exception as e:
        return {"x": [], "y": [], "error": str(e)}
