# Heart Disease Risk Prediction

Machine-learning web app that estimates heart disease risk (Low / Medium / High) from routine clinical values, and explains each result with SHAP.

**Live app:** <https://hd-cardiopredict-vqdfxlxqtzrftwf6n4yk2b.streamlit.app/>

## Features
- Random Forest model trained on the UCI Heart Disease (Cleveland) dataset
- Low / Medium / High risk level with a risk gauge
- Top contributing factors for each patient (SHAP)
- Fallback model for patients without the thalassemia test result
- Rule-based suggestions and a downloadable PDF report

## Results (test set of 60 patients)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.833 | 0.846 | 0.786 | 0.815 | 0.950 |
| SVM (linear) | 0.850 | 0.880 | 0.786 | 0.830 | 0.953 |
| Random Forest | 0.867 | 0.885 | 0.821 | 0.852 | 0.946 |

The model without the thalassemia attribute reached an accuracy of 0.833.
The test set is small (one patient is about 1.7 percentage points), so differences between models should not be over-interpreted.

## Run locally


