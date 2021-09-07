'''
After manually deleting some database files, execute to modify the JSON-configuration file of the database.
'''

from create_dataset.common import fix_json_config

fix_json_config(log_file="create_dataset/datasets/dataset.json")