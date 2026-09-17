"""
evaluate.py
Rigorous statistical evaluation of deep learning models on unseen test data.
Computes test cross-entropy loss, top-1 accuracy, top-2 accuracy, inference latency,
full classification reports, and confusion matrices.

Author: Ajnish Kumar | Roll No: 241809046713
BCA, Vinoba Bhave University, Hazaribag
Yuva Intern - Virtual Data Science with Python Trainee
September 2026
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_models(mlp_model, cnn_model, data_dict, output_dir):
    """
    Evaluates both models on the independent 10,000 sample test set.
    """
    os.makedirs(output_dir, exist_ok=True)

    X_test_mlp = data_dict["X_test_mlp"]
    X_test_cnn = data_dict["X_test_cnn"]
    y_test = data_dict["y_test"]
    class_names = data_dict["class_names"]

    # ==========================================
    # 1. Inference and Metrics: Baseline MLP
    # ==========================================
    print("[INFO] Evaluating Baseline MLP on 10,000 test images...")
    start_time = time.time()
    y_prob_mlp = mlp_model.predict(X_test_mlp, batch_size=128, verbose=0)
    mlp_infer_duration = time.time() - start_time
    y_pred_mlp = np.argmax(y_prob_mlp, axis=1)

    mlp_loss, mlp_acc = mlp_model.evaluate(X_test_mlp, y_test, batch_size=128, verbose=0)

    # Top-2 accuracy
    top2_idx_mlp = np.argsort(y_prob_mlp, axis=1)[:, -2:]
    mlp_top2_acc = np.mean([y_test[i] in top2_idx_mlp[i] for i in range(len(y_test))])

    # ==========================================
    # 2. Inference and Metrics: Regularized CNN
    # ==========================================
    print("[INFO] Evaluating Regularized CNN on 10,000 test images...")
    start_time = time.time()
    y_prob_cnn = cnn_model.predict(X_test_cnn, batch_size=128, verbose=0)
    cnn_infer_duration = time.time() - start_time
    y_pred_cnn = np.argmax(y_prob_cnn, axis=1)

    cnn_loss, cnn_acc = cnn_model.evaluate(X_test_cnn, y_test, batch_size=128, verbose=0)

    # Top-2 accuracy
    top2_idx_cnn = np.argsort(y_prob_cnn, axis=1)[:, -2:]
    cnn_top2_acc = np.mean([y_test[i] in top2_idx_cnn[i] for i in range(len(y_test))])

    # ==========================================
    # 3. Benchmark Summary Table
    # ==========================================
    benchmark_df = pd.DataFrame([
        {
            "Architecture": "Baseline MLP",
            "Test_Loss": round(mlp_loss, 4),
            "Test_Accuracy": round(mlp_acc * 100, 2),
            "Top_2_Accuracy": round(mlp_top2_acc * 100, 2),
            "Error_Rate": round((1 - mlp_acc) * 100, 2),
            "Inference_Total_Sec": round(mlp_infer_duration, 3),
            "Latency_per_1000_Img_ms": round((mlp_infer_duration / len(y_test)) * 1000 * 1000, 2),
            "Total_Params": mlp_model.count_params()
        },
        {
            "Architecture": "Deep Regularized CNN",
            "Test_Loss": round(cnn_loss, 4),
            "Test_Accuracy": round(cnn_acc * 100, 2),
            "Top_2_Accuracy": round(cnn_top2_acc * 100, 2),
            "Error_Rate": round((1 - cnn_acc) * 100, 2),
            "Inference_Total_Sec": round(cnn_infer_duration, 3),
            "Latency_per_1000_Img_ms": round((cnn_infer_duration / len(y_test)) * 1000 * 1000, 2),
            "Total_Params": cnn_model.count_params()
        }
    ])
    benchmark_path = os.path.join(output_dir, "test_evaluation_benchmark.csv")
    benchmark_df.to_csv(benchmark_path, index=False)
    print(f"\n[INFO] Test Evaluation Benchmark:\n{benchmark_df.to_string(index=False)}")

    # ==========================================
    # 4. Detailed Classification Reports
    # ==========================================
    mlp_rep_dict = classification_report(y_test, y_pred_mlp, target_names=class_names, output_dict=True)
    cnn_rep_dict = classification_report(y_test, y_pred_cnn, target_names=class_names, output_dict=True)

    df_rep_mlp = pd.DataFrame(mlp_rep_dict).transpose().reset_index().rename(columns={"index": "Class"})
    df_rep_cnn = pd.DataFrame(cnn_rep_dict).transpose().reset_index().rename(columns={"index": "Class"})

    df_rep_mlp.to_csv(os.path.join(output_dir, "classification_report_mlp.csv"), index=False)
    df_rep_cnn.to_csv(os.path.join(output_dir, "classification_report_cnn.csv"), index=False)

    # Comparative Per-Class F1 Table
    per_class_f1 = []
    for cls in class_names:
        f1_mlp = mlp_rep_dict[cls]["f1-score"]
        f1_cnn = cnn_rep_dict[cls]["f1-score"]
        delta = f1_cnn - f1_mlp
        per_class_f1.append({
            "Class_Name": cls,
            "MLP_Precision": round(mlp_rep_dict[cls]["precision"], 4),
            "MLP_Recall": round(mlp_rep_dict[cls]["recall"], 4),
            "MLP_F1": round(f1_mlp, 4),
            "CNN_Precision": round(cnn_rep_dict[cls]["precision"], 4),
            "CNN_Recall": round(cnn_rep_dict[cls]["recall"], 4),
            "CNN_F1": round(f1_cnn, 4),
            "F1_Improvement": round(delta, 4),
            "Pct_Gain": f"{((delta / f1_mlp) * 100):+.2f}%" if f1_mlp > 0 else f"+{delta * 100:.1f}%"
        })
    df_per_class = pd.DataFrame(per_class_f1)
    df_per_class.to_csv(os.path.join(output_dir, "per_class_f1_comparison.csv"), index=False)

    # ==========================================
    # 5. Confusion Matrices & Error Mining
    # ==========================================
    cm_mlp = confusion_matrix(y_test, y_pred_mlp)
    cm_cnn = confusion_matrix(y_test, y_pred_cnn)

    cm_cnn_df = pd.DataFrame(cm_cnn, index=class_names, columns=class_names)
    cm_cnn_df.to_csv(os.path.join(output_dir, "confusion_matrix_cnn.csv"))

    # Mine top misclassified pairs for CNN
    error_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_cnn[i, j] > 0:
                row_total = np.sum(cm_cnn[i, :])
                error_pct = round((cm_cnn[i, j] / row_total) * 100, 2) if row_total > 0 else 0.0
                error_pairs.append({
                    "True_Class": class_names[i],
                    "Predicted_As": class_names[j],
                    "Error_Count": int(cm_cnn[i, j]),
                    "Error_Rate_Pct": error_pct
                })
    df_errors = pd.DataFrame(error_pairs).sort_values(by="Error_Count", ascending=False).reset_index(drop=True)
    df_errors.head(10).to_csv(os.path.join(output_dir, "top_misclassified_pairs_cnn.csv"), index=False)

    return {
        "benchmark_df": benchmark_df,
        "df_rep_mlp": df_rep_mlp,
        "df_rep_cnn": df_rep_cnn,
        "df_per_class": df_per_class,
        "cm_mlp": cm_mlp,
        "cm_cnn": cm_cnn,
        "y_prob_cnn": y_prob_cnn,
        "y_pred_cnn": y_pred_cnn,
        "df_errors": df_errors
    }
