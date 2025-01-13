import sys
import os

import sys
import os
import shutil
import stat
from pathlib import Path
import logging

def validate_output_dir(output_dir):
    """
    Validates the output directory, creating it if necessary.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        
class ConfigManager:
    def __init__(self, config_filename="config.txt"):
        self.config_filename = config_filename
        self._ensure_config_exists()

    def _get_base_path(self):
        """
        Determines the appropriate base path for resources whether running as bundled or not
        """
        if getattr(sys, 'frozen', False):
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS
        else:
            # Get the directory where the script is located
            base_path = os.path.dirname(os.path.abspath(__file__))
        
        return base_path

    def _get_config_dir(self):
        """
        Returns the directory where config should be stored
        For bundled apps, uses user's home directory
        For development, uses current directory
        """
        if getattr(sys, 'frozen', False):
            # Use platform-appropriate config directory
            if sys.platform == 'darwin':
                config_dir = os.path.join(str(Path.home()), 'Library', 'Application Support', 'AI-Midi-Helper')
            elif sys.platform == 'win32':
                config_dir = os.path.join(os.getenv('APPDATA'), 'AI-Midi-Helper')
            else:  # Linux and others
                config_dir = os.path.join(str(Path.home()), '.config', 'ai-midi-helper')
        else:
            config_dir = os.getcwd()
            
        os.makedirs(config_dir, exist_ok=True)
        return config_dir

    def _ensure_config_exists(self):
        """
        Ensures config file exists, creating it if necessary
        """
        config_path = self.get_config_path()
        if not os.path.exists(config_path):
            # Copy from bundled resources if available
            bundled_config = os.path.join(self._get_base_path(), self.config_filename)
            if os.path.exists(bundled_config):
                shutil.copy2(bundled_config, config_path)
            else:
                # Create empty config if no bundled version exists
                with open(config_path, 'w') as f:
                    f.write("output=\n")  # Initialize with empty output setting

    def get_config_path(self):
        """
        Returns the full path to the config file
        """
        return os.path.join(self._get_config_dir(), self.config_filename)

    def read_config(self):
        """
        Reads and returns config as dictionary
        """
        config = {}
        try:
            with open(self.get_config_path(), 'r') as f:
                for line in f:
                    if '=' in line:
                        key, value = line.strip().split('=', 1)
                        config[key.strip()] = value.strip()
        except Exception as e:
            print(f"Error reading config: {e}")
        return config

    def set_config(self, key, value):
        """
        Sets or updates a config value
        """
        config = self.read_config()
        config[key] = value
        
        try:
            with open(self.get_config_path(), 'w') as f:
                for k, v in config.items():
                    f.write(f"{k}={v}\n")
        except Exception as e:
            raise IOError(f"Failed to write config: {e}")

    def get_output_directory(self):
        """
        Gets the output directory from config, ensuring it exists
        """
        config = self.read_config()
        output_dir = config.get('output', '')
        
        if not output_dir or output_dir == '':
            raise ValueError("Output directory not set in config")
            
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

def set_executable_permissions(path):
    """
    Sets executable permissions on a file
    """
    try:
        current_mode = os.stat(path).st_mode
        os.chmod(path, current_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as e:
        raise PermissionError(f"Failed to set execute permissions for {path}: {e}")
    
def get_output():
    config_manager = ConfigManager()
    try:
        return config_manager.get_output_directory()
    except ValueError as e:
        print(str(e))
        return ""
 
def set_permissions(path):
    """Ensure the file has execute permissions."""
    import stat
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    except Exception as e:
        raise PermissionError(f"Failed to set execute permissions for {path}: {e}")



def get_data_path(filename):
    """
    Retrieves the absolute path to a resource file, adapting for py2app.
    """
    if getattr(sys, 'frozen', False):  # Check if bundled with PyInstaller
        print("Frozen")
        # Resources in PyInstaller are located in the same directory as the executable
        base_path = sys._MEIPASS  # This is set by PyInstaller to point to the temp directory containing bundled files
        print(base_path)
        base_path = os.path.join(base_path, filename)
    else:
        print("Not frozen")
        # Fallback to the script's directory when not bundled
        base_path = os.path.join(os.path.dirname(__file__), "..", "..", filename)
    return base_path  # Removed the comma to return a string



def set_config(key, value, config_path="config.txt"):
    """
    Sets a key-value pair in config.txt. If the key exists, it updates the value; otherwise, it adds the key.
    """
    config_path = get_data_path(config_path)
    config = read_config(config_path)
    config[key] = value
    with open(config_path, "w") as f:
        for k, v in config.items():
            f.write(f"{k}={v}\n")

def read_config(config_path="config.txt"):
    """
    Reads the config.txt and returns a dictionary of configurations.
    """
    config = {}
    config_path = get_data_path(config_path)
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            for line in f:
                if '=' in line:
                    key, value = line.strip().split('=', 1)
                    config[key.strip()] = value.strip()
    return config

def get_config():
    """
    Retrieves the entire configuration as a dictionary.
    """
    return read_config()

def get_embedded_python_path():
    """Get the system Python path."""
    import shutil
    try:
        # First try using 'python3' command
        python_path = shutil.which('python3')
        if python_path:
            return python_path
            
        # Fall back to 'python' if python3 not found
        python_path = shutil.which('python')
        if python_path:
            return python_path
            
        # If on macOS, try the common Python locations
        if sys.platform == 'darwin':
            common_paths = [
                '/usr/bin/python3',
                '/usr/local/bin/python3',
                '/opt/homebrew/bin/python3'
            ]
            for path in common_paths:
                if os.path.exists(path):
                    return path
                    
        raise FileNotFoundError("Could not find Python interpreter")
        
    except Exception as e:
        raise Exception(f"Error finding Python interpreter: {str(e)}")