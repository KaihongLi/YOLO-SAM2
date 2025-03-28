"""
Created on Wed Mar 19 2025 by LKH
Logging BaseConfig
"""
import json
import logging

# output to both console and file
# handler_file = logging.FileHandler('./logs/yolo_sam2.txt')
# handler_file.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler_file.setFormatter(formatter)
# handler_stream = logging.StreamHandler()
# handler_stream.setLevel(logging.INFO)
# handler_stream.setFormatter(formatter)
# logging.basicConfig(level=logging.INFO, handlers=[handler_file, handler_stream])

# output to the console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class Logger:
    def __init__(self):
        self.entries = {}

    def add_entry(self, entry):
        self.entries[len(self.entries) + 1] = entry

    def __str__(self):
        return json.dumps(self.entries, sort_keys=True, indent=4)
