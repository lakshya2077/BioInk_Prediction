import logging
import os

# --- Logging Utilities ---
def setup_logging(log_path="degradation_project.log"):
    """
    Sets up logging to a file.
    """
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def log_prediction(input_data: dict, prediction: dict):
    """
    Logs input and prediction to the log file.
    """
    logging.info(f"INPUT: {input_data}")
    logging.info(f"PREDICTION: {prediction}")


# --- Input Validation Utilities ---
def validate_input(input_data: dict) -> tuple[bool, str]:
    """
    Validates that input values are within reasonable physical/experimental ranges.
    Returns (is_valid, error_message).
    """
    try:
        if input_data['Porosity_Percentage'] < 0 or input_data['Porosity_Percentage'] > 100:
            return False, "Porosity (%) must be between 0 and 100."
        if input_data['Immersion_Time_Days'] < 0:
            return False, "Immersion time must be non-negative."
        if input_data['Mechanical_Loading'] not in [0, 1]:
            return False, "Mechanical loading must be 0 (No) or 1 (Yes)."
        if not input_data['Scaffold_Geometry']:
            return False, "Scaffold geometry cannot be empty."
    except KeyError as e:
        return False, f"Missing input key: {e}"

    return True, ""
