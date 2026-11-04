import os
import sys

# Make the project root importable (engine, runner, gui, ...) when running pytest from anywhere
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
