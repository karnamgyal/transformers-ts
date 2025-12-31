# YAML Configuration Loader
import yaml

# Function to load a YAML configuration file
def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
