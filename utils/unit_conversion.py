import os

# Conversion constants
C = 299792458  # Speed of light in m/s
H = 4.135667696e-15  # Planck's constant in eV·s
EV_TO_NM = 1239.84198300923944  # Conversion factor for eV to nm

def convert_from_nm(value_nm, target_unit):
    """Convert a value in nanometers to the target unit."""
    try:
        conversion_factors = {
            "nm": value_nm,
            "µm": value_nm * 1e-3,
            "eV": EV_TO_NM / value_nm,
            "meV": (EV_TO_NM / value_nm) * 1e3,
            "THz": C / (value_nm * 1e-9) / 1e12,
            "fs": value_nm * 3.3356409519815204 * 1e-3,
            "ps": value_nm * 3.3356409519815204 * 1e-6,
            "cm⁻¹": 1e7 / value_nm
        }
        return conversion_factors.get(target_unit, None)
    except ZeroDivisionError:
        return 0

def convert_units(input_unit, input_value):
    """Convert a given input unit and value to all possible target units."""
    if input_unit == "nm":
        value_in_nm = input_value
    elif input_unit == "µm":
        value_in_nm = input_value * 1e3
    elif input_unit == "eV":
        value_in_nm = EV_TO_NM / input_value
    elif input_unit == "meV":
        value_in_nm = EV_TO_NM / (input_value * 1e-3)
    elif input_unit == "THz":
        value_in_nm = C / (input_value * 1e12) * 1e9
    elif input_unit == "fs":
        value_in_nm = input_value * 3.3356409519815204
    elif input_unit == "ps":
        value_in_nm = input_value * 3.3356409519815204 * 1e3
    elif input_unit == "cm⁻¹":
        value_in_nm = 1e7 / input_value
    else:
        return None

    return {unit: convert_from_nm(value_in_nm, unit) for unit in ["nm", "µm", "eV", "meV", "THz", "fs", "ps", "cm⁻¹"]}
