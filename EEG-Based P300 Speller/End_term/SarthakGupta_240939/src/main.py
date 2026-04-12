import warnings
warnings.filterwarnings("ignore")
import numpy as np
from dataloader import load_all
from preprocess import extract_epochs
from features import extract_features
from models import train_model,decode_string
from evaluate import print_comparison,char_accuracy,calculate_itr,classification_metrics,stratified_kfold_cv,plot_confusion_matrix

TRUE_STRINGS={
    "Subject_A_Test.mat":"WQXPLZCOMRKO97YFZDEZ1DPI9NNVGRQDJCUVRMEUOOOJD2UFYPOO6J7LDGYEGOA5VHNEHBTXOO1TDOILUEE5BFAEEXAW_K4R3MRU",
    "Subject_B_Test.mat":"MERMIROOMUHJPXJOHUVLEORZP3GLOO7AUFDKEFTWEOOALZOP9ROCGZET1Y19EWX65QUYU7NAK_4YCJDVDNGQXODBEV2B5EFDIDNR"
}

MODE="lda" #for Lda: "lda", for svm: "svm", for eegnet: "eegnet"
SHOW_UI=False
UI_REPS=5

def run():
    datasets=load_all("data")
    ui_results={}

    for subject in ["A","B"]:
        print(f"\n========== SUBJECT {subject} | {MODE.upper()} ==========")
        train_key=f"Subject_{subject}_Train.mat"
        d=datasets[train_key]

        segs,labels,codes=extract_epochs(d["signal"],d["flashing"],d["stimulus"],d["stim_type"])
        y_all=np.concatenate(labels)
        X_train,X_per_char_train=extract_features(segs,y_all=y_all,fit=True,mode=MODE)

        n_chans=X_train.shape[1] if MODE=="eegnet" else None
        n_times=X_train.shape[2] if MODE=="eegnet" else None

        print("X_train shape:",X_train.shape)

        model=train_model(X_train,y_all,model_type=MODE,n_chans=n_chans,n_times=n_times)
        print("Model trained.")

        if MODE!="eegnet":
            print("\n--- Epoch-level Metrics ---")
            classification_metrics(model,X_train,y_all)
            print()
            stratified_kfold_cv(model,X_train,y_all,k=5)
            plot_confusion_matrix(model,X_train,y_all,subject=subject,mode=MODE)

        test_key=f"Subject_{subject}_Test.mat"
        d=datasets[test_key]

        segs_test,_,codes_test=extract_epochs(d["signal"],d["flashing"],d["stimulus"])
        X_per_char_test=extract_features(segs_test,fit=False,mode=MODE)

        print("\n--- Character Decoding ---")

        for n_reps in [3,5,7,10,15]:
            pred=decode_string(model,X_per_char_test,codes_test,n_reps=n_reps)
            true=TRUE_STRINGS[test_key]
            acc=char_accuracy(pred,true)
            print(f"  reps={n_reps}",end=" | ")
            calculate_itr(acc,n_reps=n_reps)

        ui_results[subject]=decode_string(model,X_per_char_test,codes_test,n_reps=UI_REPS)

    if SHOW_UI:
        from speller_ui import launch_speller
        launch_speller(ui_results, repetitions=UI_REPS)

if __name__=="__main__":
    run()
