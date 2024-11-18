#!/bin/bash

# Exit on any error
set -e

echo "Starting script execution..."

# Run scripts sequentially
echo "Running link_litigation.py..."
python3 link_litigation.py

echo -e "\nRunning combine_registrations.py..."
python3 combine_registrations.py

echo -e "\nRunning link_registrations.py..."
python3 link_registrations.py

echo -e "\nAll scripts completed successfully!"
