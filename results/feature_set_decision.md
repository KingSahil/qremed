# Feature Set Decision — Q-REMED Phase 1

**Primary feature set:** 4_features (4 features)
- worst concave points
- worst perimeter
- mean concave points
- worst radius

**Backup feature set:** 6_features (6 features)
- worst concave points
- worst perimeter
- mean concave points
- worst radius
- mean perimeter
- worst area

## Rationale
4-feature subset maintains sensitivity/specificity within the pre-registered thresholds of the 6-feature subset (sensitivity drop=0.0000, specificity drop=0.0139, thresholds=0.02/0.05), so the smaller (4-qubit) set is preferred per the project's 'smallest useful circuit' bias.

## Average metrics by subset (across both classical models)

| Subset | Sensitivity | Specificity | F1 | Accuracy |
|---|---|---|---|---|
| 4_features | 0.9405 | 0.9236 | 0.9080 | 0.9298 |
| 6_features | 0.9405 | 0.9375 | 0.9188 | 0.9386 |
