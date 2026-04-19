import warnings
warnings.filterwarnings("ignore")
import os
import numpy as np
from src.load_data import load_subject
from src.preprocess import extract_epochs_per_char
from src.features import transform_epochs
from src.model import get_model
from src.decode import decode_with_reps
from src.evaluate import (char_accuracy, classification_metrics, stratified_kfold_cv,
                           plot_confusion_matrix, calculate_itr,
                           plot_accuracy_vs_reps, plot_itr_vs_reps, char_confusion)

TRUE_STRINGS = {
    "Subject_A_Test.mat": "WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU",
    "Subject_B_Test.mat": "MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR"
}

model_type = "svm"  # "lda" | "svm" | "eegnet"
UI_REPS    = 5
SHOW_UI    = False
ui_results = {}

os.makedirs("results", exist_ok=True)
results_log = open(f"results/{model_type}_results.txt", "a")

def log(msg):
    print(msg)
    results_log.write(msg + "\n")
    results_log.flush()

for subject in ["A", "B"]:
    log(f"\n===== SUBJECT {subject} | {model_type.upper()} =====")

    signal, flashing, stim_code, stim_type, _ = load_subject(
        f"data/BCI_Comp_III_Wads_2004/Subject_{subject}_Train.mat"
    )
    segments, labels, _ = extract_epochs_per_char(signal, flashing, stim_code, stim_type)
    y_all = np.concatenate(labels)
    X_train, _ = transform_epochs(segments, y_all=y_all, fit=True, mode=model_type)

    model = get_model(model_type, X_train.shape[1], X_train.shape[2]) if model_type == "eegnet" else get_model(model_type)
    model.fit(X_train, y_all)
    log("Trained.")

    if model_type != "eegnet":
        log("Metrics based on training data")
        classification_metrics(model, X_train, y_all)
        stratified_kfold_cv(model, X_train, y_all)
        plot_confusion_matrix(
            model, X_train, y_all,
            save_path=f"results/{model_type}_Subject{subject}_epoch_confusion.png"
        )

    signal, flashing, stim_code, _, _ = load_subject(
        f"data/BCI_Comp_III_Wads_2004/Subject_{subject}_Test.mat"
    )
    segments, _, codes = extract_epochs_per_char(signal, flashing, stim_code)
    X_per_char_test = transform_epochs(segments, mode=model_type)

    reps_list, acc_list, itr_list = [], [], []
    best_pred_string = ""

    for n_reps in [3, 5, 7, 10, 15]:
        pred_chars = [decode_with_reps(model.predict_proba(X), c, n_reps)
                      for X, c in zip(X_per_char_test, codes)]
        pred_string = "".join(pred_chars)
        test_key = f"Subject_{subject}_Test.mat"
        log(f"reps={n_reps} | {pred_string}")
        acc = char_accuracy(pred_string, TRUE_STRINGS[test_key])
        itr = calculate_itr(acc, n_reps=n_reps)

        reps_list.append(n_reps)
        acc_list.append(acc)
        itr_list.append(itr)

        if n_reps == UI_REPS:
            ui_results[subject] = pred_string
            best_pred_string = pred_string

    plot_accuracy_vs_reps(
        subject, reps_list, acc_list,
        save_path=f"results/{model_type}_Subject{subject}_accuracy_vs_reps.png"
    )
    plot_itr_vs_reps(
        subject, reps_list, itr_list,
        save_path=f"results/{model_type}_Subject{subject}_itr_vs_reps.png"
    )
    char_confusion(
        best_pred_string, TRUE_STRINGS[f"Subject_{subject}_Test.mat"],
        save_path=f"results/{model_type}_Subject{subject}_char_confusion.png"
    )

results_log.close()

if SHOW_UI:
    from src.speller_ui import run_speller
    run_speller(ui_results, n_reps=UI_REPS)