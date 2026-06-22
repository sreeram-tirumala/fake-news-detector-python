import joblib, os, glob

def save_artifact(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_artifact(path):
    return joblib.load(path)

def newest_artifact(pattern):
    matches = glob.glob(pattern)
    return max(matches, key=os.path.getmtime) if matches else None
