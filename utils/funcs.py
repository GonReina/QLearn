import json
from pathlib import Path
import jsonschema

# 1. Define the Project Root
PROJECT_ROOT = Path(__file__).parent.parent

# 2. Define the path to the config folder
CONFIG_DIR = PROJECT_ROOT / "configs"

# Schema describing the expected structure and types for config.json
CONFIG_SCHEMA = {
    "type": "object",
    "properties": {
        "simulation_settings": {
            "type": "object",
            "properties": {
                "output_dir": {"type": "string"},
                "solver_method": {"type": "string"},
                "rtol": {"type": "number"},
                "atol": {"type": "number"},
                "num_cpus": {"type": "integer"},
                "sim_length": {"type": "integer"},
                "usteps": {"type": "integer"}
            },
            "required": ["sim_length", "usteps", "solver_method", "rtol", "atol"]
        },
        "system_parameters": {
            "type": "object",
            "properties": {
                "N": {"type": "integer", "minimum": 1},
                "n_max_transmon": {"type": "integer", "minimum": 1},
                "n_max_resonator": {"type": "integer", "minimum": 1}
            },
            "required": ["N", "n_max_transmon", "n_max_resonator"]
        },
        "physical_constants": {
            "type": "object",
            "properties": {
                "eta": {"type": "number"},
                "phiq": {"type": "number"},
                "phia": {"type": ["string", "number"]},
                "J": {"type": ["string", "number"]},
                "nu": {"type": ["string", "number"]},
                "delta": {"type": ["string", "number"]},
                "de": {"type": "number"},
                "wq": {"type": ["string", "number"]},
                "EJ": {"type": ["string", "number"]},
                "kappa": {"type": "number"}
            },
            "required": ["eta", "phiq", "phia", "J", "nu", "delta", "de", "wq", "EJ", "kappa"]
        }
    },
    "required": ["simulation_settings", "system_parameters", "physical_constants"]
}



def load_params(filename: str = "simulation_params.json") -> dict:
    """
    Loads and validates parameters from a JSON configuration file.
    
    Raises:
        jsonschema.ValidationError: If the config file does not match the schema.
        FileNotFoundError: If the config file cannot be found.
    """
    config_path = CONFIG_DIR / filename

    with open(config_path, 'r', encoding="utf8") as f:
        config = json.load(f)
    
    # Check if the config file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    # Validate the loaded configuration against the schema
    jsonschema.validate(instance=config, schema=CONFIG_SCHEMA)

    # Combine all parameter sections into a single dictionary
    params = {**config['simulation_settings'], **config['system_parameters'], **config['physical_constants']}
    
    return params