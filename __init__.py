import os
import sys

# Get the absolute path of the current directory (i.e., the directmhp folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Add this directory to sys.path if it's not already present, so absolute imports work
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
