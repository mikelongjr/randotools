#!/usr/bin/env python3
import subprocess
import time
import json
import shutil
import re

def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def get_gpu_temperature_rocm():
    """Retrieves the AMD GPU temperature using rocm-smi."""
    try:
        # We target device 0, but this could be parameterized.
        output = subprocess.check_output(['rocm-smi', '-d', '1', '--showtemp', '--json'])
        data = json.loads(output)

        # The JSON structure is typically like: {"card0": {"Temperature (C)": "55.0"}}
        # We find the first card key (e.g., 'card0').
        card_key = next(iter(data))
        card_data = data[card_key]

        temp_str = None
        # rocm-smi output for temperature is not consistent across versions/cards.
        # Try a few known keys in order of preference.
        possible_keys = [
            'Temperature (Sensor edge) (C)',
            'Temperature (Sensor junction) (C)',
            'Temperature (C)',
        ]
        for key in possible_keys:
            if key in card_data:
                temp_str = card_data[key]
                break

        # If none of the specific keys are found, try a more general search.
        if not temp_str:
            for key, value in card_data.items():
                if 'temp' in key.lower() and '(c)' in key.lower():
                    temp_str = value
                    break

        if temp_str:
            # The value can be "55.0" or "55.0c". We need to extract the number.
            match = re.match(r'\s*([0-9.]+)', temp_str)
            if match:
                return int(float(match.group(1)))

        # If we're here, we couldn't find or parse the temperature.
        print("Error: Could not find or parse a temperature value from rocm-smi.")
        print(f"  Available keys in '{card_key}': {list(card_data.keys())}")
        return None
    except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError, StopIteration, KeyError, ValueError) as e:
        print(f"Error getting GPU temperature: {e}")
        return None

def set_power_limit_rocm(limit):
    """
    Sets the GPU power limit in Watts using rocm-smi.
    NOTE: This command usually requires root privileges (sudo).
    """
    try:
        # We target device 0, but this could be parameterized.
        subprocess.run(['sudo', 'rocm-smi', '-d', '1', '--setpower', str(limit)], check=True, capture_output=True)
        print(f"Power limit set to {limit}W")
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to set power limit. Do you have sudo permissions? Error: {e}")

def main():
    # --- Configuration ---
    # These are example values. Adjust them for your specific GPU.
    # You can find your GPU's default/max power limit with `rocm-smi -P`
    DEFAULT_POWER_LIMIT = 125       # Watts
    HIGH_TEMP_POWER_LIMIT = 90     # Watts
    CRITICAL_TEMP_POWER_LIMIT = 60# Watts

    HIGH_TEMP_THRESHOLD = 85        # Celsius
    CRITICAL_TEMP_THRESHOLD = 95    # Celsius
    POLL_INTERVAL = 2               # seconds
    # ---------------------

    if not is_tool('rocm-smi'):
        print("Error: 'rocm-smi' command not found. Please ensure ROCm is installed and in your PATH.")
        return

    print("Starting GPU temperature watcher for AMD GPU (using rocm-smi)...")
    print(f"NOTE: Setting power limits requires sudo privileges. You may be prompted for your password.")

    try:
        while True:
            temp = get_gpu_temperature_rocm()
            if temp is not None:
                print(f"GPU Temperature: {temp}°C")
                if temp > CRITICAL_TEMP_THRESHOLD:
                    print("CRITICAL TEMPERATURE! Reducing power limit drastically.")
                    set_power_limit_rocm(CRITICAL_TEMP_POWER_LIMIT)
                elif temp > HIGH_TEMP_THRESHOLD:
                    print("High temperature. Reducing power limit.")
                    set_power_limit_rocm(HIGH_TEMP_POWER_LIMIT)
                else:
                    print("Temperature OK. Setting power limit to default.")
                    set_power_limit_rocm(DEFAULT_POWER_LIMIT)
            else:
                print("Could not retrieve GPU temperature. Will try again.")

            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        print("\nExiting. Restoring default power limit...")
        set_power_limit_rocm(DEFAULT_POWER_LIMIT)
        print("Done.")

if __name__ == '__main__':
    main()
