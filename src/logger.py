import json
from datetime import datetime
import os

LOG_FILE = 'history.json'

def log_analysis(filename, object_counts, scene_description, driver_recommendations, input_type):
    data = {
        "timestamp" : datetime.now().strftime("%Y-%m-%d %H-%M-%S"),
        "filename" : filename,
        "object_counts" : object_counts,
        "scene_description" : scene_description,
        "driver_recommendations" : driver_recommendations,
        "input_type": input_type
    }

    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            history = json.load(f)

    else:
        history = []

    history.append(data)

    with open(LOG_FILE, "w") as f:
        json.dump(history, f, indent=4)