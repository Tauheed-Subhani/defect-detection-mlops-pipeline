import numpy as np
from PIL import Image
import os
import json
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

def extract_features(image_path):
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    arr = np.array(img) / 255.0
    return {
        'mean_r': float(arr[:,:,0].mean()),
        'mean_g': float(arr[:,:,1].mean()),
        'mean_b': float(arr[:,:,2].mean()),
        'std_r': float(arr[:,:,0].std()),
        'std_g': float(arr[:,:,1].std()),
        'std_b': float(arr[:,:,2].std()),
        'brightness': float(arr.mean())
    }

def load_features_from_folder(folder, limit=100):
    features = []
    count = 0
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(root, file)
                features.append(extract_features(path))
                count += 1
                if count >= limit:
                    return features
    return features

print("Loading training data...")
train_features = load_features_from_folder('archive/casting_data/casting_data/train', limit=200)
train_df = pd.DataFrame(train_features)

print("Loading production data...")
test_features = load_features_from_folder('production_stream', limit=200)
test_df = pd.DataFrame(test_features)

print("Running drift analysis...")
report = Report([DataDriftPreset()])
my_eval = report.run(train_df, test_df)
my_eval.save_html('drift_report.html')

# Extract drift results so other scripts (retrain_trigger.py) can read the outcome
result_dict = my_eval.dict()

drift_count = result_dict['metrics'][0]['value']['count']
drift_share = result_dict['metrics'][0]['value']['share']
dataset_drift = drift_share >= 0.5  # matches Evidently's own threshold shown in the HTML report

drift_status = {
    "dataset_drift": dataset_drift,
    "drift_share": drift_share,
    "drifted_column_count": drift_count
}

with open('drift_status.json', 'w') as f:
    json.dump(drift_status, f, indent=2)

print("Drift report saved to drift_report.html")
print(f"Drift status saved to drift_status.json: {drift_status}")