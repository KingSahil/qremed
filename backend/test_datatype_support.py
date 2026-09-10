"""
Verification script for universal datatype support and target auto-detection in Q-REMED.
"""
import io
import pandas as pd
import numpy as np

from app.engine.dataset_analysis import (
    detect_target_candidate, get_column_metadata, classify_columns,
    validate_dataset, analyze_dataset, get_target_nature
)
from app.engine.preprocessing import (
    leakage_safe_preprocess, clean_feature_series, expand_datetime_columns, select_usable_features
)
from app.engine.classical_models import train_all_baselines
from app.engine.evaluation import evaluate_sklearn_model

HEART_FAILURE_CSV = """age,anaemia,creatinine_phosphokinase,diabetes,ejection_fraction,high_blood_pressure,platelets,serum_creatinine,serum_sodium,sex,smoking,time,DEATH_EVENT
75,0,582,0,20,1,265000,1.9,130,1,0,4,1
55,0,7861,0,38,0,263358.03,1.1,136,1,0,6,1
65,0,146,0,20,0,162000,1.3,129,1,1,7,1
50,1,111,0,20,0,210000,1.9,137,1,0,7,1
65,1,160,1,20,0,327000,2.7,116,0,0,8,1
90,1,47,0,40,1,204000,2.1,132,1,1,8,1
75,1,246,0,15,0,127000,1.2,137,1,0,10,1
60,1,315,1,60,0,454000,1.1,131,1,1,10,1
65,0,157,0,65,0,263358.03,1.5,138,0,0,10,1
80,1,123,0,35,1,388000,9.4,133,1,1,10,1
75,1,81,0,38,1,368000,4,131,1,1,10,1
62,0,231,0,25,1,253000,0.9,140,1,1,10,1
45,1,981,0,30,0,136000,1.1,137,1,0,11,1
50,1,168,0,38,1,276000,1.1,137,1,0,11,1
49,1,80,0,30,1,427000,1,138,0,0,12,0
82,1,379,0,50,0,47000,1.3,136,1,0,13,1
87,1,149,0,38,0,262000,0.9,140,1,0,14,1
45,0,582,0,14,0,166000,0.8,127,1,0,14,1
70,1,125,0,25,1,237000,1,140,0,0,15,1
48,1,582,1,55,0,87000,1.9,121,0,0,15,1
65,1,52,0,25,1,276000,1.3,137,0,0,16,0
65,1,128,1,30,1,297000,1.6,136,0,0,20,1
68,1,220,0,35,1,289000,0.9,140,1,1,20,1
53,0,63,1,60,0,368000,0.8,135,1,0,22,0
75,0,582,1,30,1,263358.03,1.83,134,0,0,23,1
80,0,148,1,38,0,149000,1.9,144,1,1,23,1
95,1,112,0,40,1,196000,1,138,0,0,24,1
70,0,122,1,45,1,284000,1.3,136,1,1,26,1
58,1,60,0,38,0,153000,5.8,134,1,0,26,1
82,0,70,1,30,0,200000,1.2,132,1,1,26,1
94,0,582,1,38,1,263358.03,1.83,134,1,0,27,1
85,0,23,0,45,0,360000,3,132,1,0,28,1
50,1,249,1,35,1,319000,1,128,0,0,28,1
50,1,159,1,30,0,302000,1.2,138,0,0,29,0
65,0,94,1,50,1,188000,1,140,1,0,29,1
69,0,582,1,35,0,228000,3.5,134,1,0,30,1
90,1,60,1,50,0,226000,1,134,1,0,30,1
82,1,855,1,50,1,321000,1,145,0,0,30,1
60,0,2656,1,30,0,305000,2.3,137,1,0,30,0
60,0,235,1,38,0,329000,3,142,0,0,30,1"""

def test_heart_failure_auto_detection():
    print("=== TEST 1: Heart Failure Auto-Detection ===")
    df = pd.read_csv(io.StringIO(HEART_FAILURE_CSV))
    suggested = detect_target_candidate(df)
    print(f"Detected target candidate: {suggested}")
    assert suggested == "DEATH_EVENT", f"Expected DEATH_EVENT, got {suggested}"
    
    meta = get_column_metadata(df)
    target_meta = [m for m in meta if m["name"] == "DEATH_EVENT"][0]
    assert target_meta["is_candidate_target"] is True
    assert target_meta["category"] == "binary"
    print("[OK] Test 1 passed: DEATH_EVENT correctly identified as target candidate.")


def test_heart_failure_binary_target():
    print("\n=== TEST 2: Native Binary Target (DEATH_EVENT) ===")
    df = pd.read_csv(io.StringIO(HEART_FAILURE_CSV))
    overview = analyze_dataset(df, "DEATH_EVENT")
    assert overview["validation"]["supported"] is True, f"Validation failed: {overview['validation']['reasons']}"
    assert overview["n_classes"] == 2
    
    prep = leakage_safe_preprocess(df, "DEATH_EVENT", test_size=0.2, random_state=42)
    assert len(prep["X_train"]) > 0
    assert prep["info"]["positive_class_name"] in ("0", "1")
    print(f"[OK] Test 2 passed: Preprocessed DEATH_EVENT. X_train shape: {prep['X_train'].shape}")


def test_heart_failure_continuous_target_age():
    print("\n=== TEST 3: Continuous Numeric Target (age - 47 classes) ===")
    df = pd.read_csv(io.StringIO(HEART_FAILURE_CSV))
    assert df["age"].nunique() > 10
    
    # 1. Analyze with default median binarization
    overview = analyze_dataset(df, "age")
    assert overview["validation"]["supported"] is True, f"Continuous target validation failed: {overview['validation']['reasons']}"
    assert overview["target_info"]["is_binarized"] is True
    assert overview["target_info"]["strategy"] == "median"
    print(f"Continuous target analysis: strategy={overview['target_info']['strategy']}, threshold={overview['target_info']['threshold']}")
    print(f"Class counts: {overview['class_counts']}")
    
    # 2. Preprocess with median binarization
    prep = leakage_safe_preprocess(df, "age", test_size=0.25, random_state=42, binarize_strategy="median")
    assert prep["y_train"].nunique() == 2
    assert prep["y_test"].nunique() == 2
    assert len(prep["info"]["feature_columns"]) == 12  # 13 total - age = 12 features
    print(f"Label map: {prep['info']['label_map']}")
    
    # 3. Fit classical model on the continuous target's binarized outcome
    models = train_all_baselines(prep["X_train"], prep["y_train"], random_state=42)
    rf = models["Random Forest"]["model"]
    ev = evaluate_sklearn_model(rf, prep["X_test"], prep["y_test"], prep["info"]["positive_label"], prep["info"]["negative_label"])
    print(f"Random Forest Accuracy on age>=threshold: {ev['metrics']['accuracy']:.3f}, Sensitivity: {ev['metrics']['sensitivity']:.3f}")
    assert not np.isnan(ev["metrics"]["accuracy"])
    print("[OK] Test 3 passed: age successfully binarized and trained with 0 errors.")


def test_mixed_datatypes_feature_encoding():
    print("\n=== TEST 4: Mixed Feature Datatypes (Categorical, Boolean, String-Numbers, Dates) ===")
    np.random.seed(42)
    n = 60
    data = {
        "patient_id": [f"ID_{i:04d}" for i in range(n)],
        "gender": np.random.choice(["Male", "Female"], size=n),
        "smoker": np.random.choice([True, False], size=n),
        "income": [f"${np.random.randint(30, 100)},{np.random.randint(100, 999)}" for _ in range(n)],
        "visit_date": pd.date_range("2023-01-01", periods=n, freq="D").astype(str),
        "blood_pressure": np.random.normal(120, 15, size=n).round(1),
        "severity": np.random.choice(["Mild", "Moderate", "Severe"], size=n),
    }
    df = pd.DataFrame(data)
    
    # Target is multiclass ("severity"), features include ID, categoricals, booleans, string-numbers, datetimes
    overview = analyze_dataset(df, "severity", positive_class="Severe")
    assert overview["validation"]["supported"] is True
    assert overview["target_info"]["target_nature"] == "multiclass"
    print(f"Multiclass OvR class counts: {overview['class_counts']}")
    
    prep = leakage_safe_preprocess(df, "severity", test_size=0.2, random_state=42, positive_class="Severe")
    assert "patient_id" in prep["info"]["dropped_id_columns"]
    assert len(prep["info"]["dropped_categorical_columns"]) == 0  # no categoricals dropped!
    assert len(prep["info"]["feature_columns"]) > 4  # one-hot + date parts + numeric
    print(f"Features after leakage-safe encoding: {prep['info']['feature_columns']}")
    
    # Train classical model
    models = train_all_baselines(prep["X_train"], prep["y_train"], random_state=42)
    assert "Logistic Regression" in models
    assert "Random Forest" in models
    print("[OK] Test 4 passed: All feature datatypes encoded and model trained.")


if __name__ == "__main__":
    test_heart_failure_auto_detection()
    test_heart_failure_binary_target()
    test_heart_failure_continuous_target_age()
    test_mixed_datatypes_feature_encoding()
    print("\nALL TESTS PASSED SUCCESSFULLY!")
