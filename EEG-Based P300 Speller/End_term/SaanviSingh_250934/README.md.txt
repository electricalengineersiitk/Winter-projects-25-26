#PURPOSE-TO BUILD EEG P300 SPELLER

##ENVIRONMENT SETUP
Environment Setup
Python Version
•	Python 3.9 or higher (3.10 recommended)
•	Use a virtual environment:
python -m venv eeg_env
source eeg_env/bin/activate   # Linux / macOS
eeg_env\Scripts\activate      # Windows

##CORE PYTHON LIBRARIES REQUIRED
Library	Version	Purpose
mne	>=1.6	EEG loading, filtering, epoching, ICA
numpy	>=1.24	Array operations and signal math
scipy	>=1.11	Bandpass filters, FFT, stats
scikit-learn	>=1.3	LDA, SVM, cross-validation, metrics
moabb	>=0.5	BCI dataset download and benchmarking
torch	>=2.0	Deep learning (EEGNet, CNN models)
matplotlib	>=3.7	Plotting ERP waveforms and results
seaborn	>=0.12	Heatmaps and confusion matrices
psychopy	>=2023.1	Visual stimulus delivery interface
pandas	>=2.0	Data management and result logging

##DATASET USED
from moabb.datasets import BNCI2014_009


##UNDERSTANDING THE DATA STRUCTURE
sessions = {
    1: {                          ← subject 1
        "session_0": {            ← session name
            "run_0": raw_object,  ← run 1
            "run_1": raw_object,  ← run 2
            "run_2": raw_object   ← run 3
        }
    }
}
It's a dictionary inside a dictionary inside a dictionary.


##PROJECT STRUCTURE
  ├── data/              # raw EEG file
  │   ├── preprocess.py  # filtering, epoching, artifact removal
  │   ├── features.py    # feature extraction functions
  │   ├── models.py      # classifiers (LDA, SVM, EEGNet)
  │   ├── evaluate.py    # cross-validation, ITR calculation
  │   └── speller_ui.py  # optional: Psychopy stimulus interface
  ├── results/           # saved models, plots, metrics

##Stage-1:Signal Preprocessing
  EEG signals typically have frequency .1 to 20 Hz. We are using lfreq=.1 and hfreq=30. This provides larger window and ensures we do not miss out on any EEG signal.
  Since power line frequency in India is 50 Hz, we are using notch filter at 50 Hz.A notch filter is a type of filter that strongly attenuates (reduces) a very narrow band of frequencies in a signal while leaving most other frequencies almost unchanged.
  Re-referencing: This is important after removing the bad channels as in EEG we look at potential difference and by re referencing to average we set our reference at the average of all electrodes which evens out any possible errors that might have crept in had we used a single electrode as our reference.

##Stage-2: Epoching
  The P300 signal occurs typically 300 ms post stimulus. By using the time window of -200 ms to 800 ms around the stimulus, we provide ourselves a wide enough window and ensure we don't miss out any potential EEG signal.
  We apply baseline correction from -200 to 0 ms. This averages the signal in the -200 to 0 ms window and then subtracts it from the entire -200 to 800 ms window. This removes any slow drifts which is essential for it to become M.L. ready. Trade off in baseline correction:
If the -200 to 0ms window happens to contain:
A small eye movement
A muscle twitch
Random noise spike
Then subtracting it will inject that artifact into the entire epoch — distorting the very signal we want to analyze.
However, here's why we still do it:
1. The window is very short (200ms)
Probability of a major artifact in just 200ms is low
Major artifacts are usually caught by reject=dict(eeg=100e-6) anyway — those epochs get dropped entirely
2. The alternative is worse
Without baseline correction, slow drifts and DC offsets make epochs incomparable to each other
Two epochs of the same stimulus might look completely different just because of drift and the classifier will get confused.

##Stage 3- Feature Extraction
Why XDawn?P300 is an EVENT RELATED POTENTIAL
	→ brain responds to a specific stimulus at a specific time
	→ Xdawn finds filters that maximize signal-to-noise ratio of this time-locked response
	  CSP doesn't care about timing and would find nothing useful. CSP is suitable for motor imagery tasks where time window does not 	  matter.
XDawn ledoit_wolf: Average rereferencing reduces rank of our data by 1. Xdawn tries to invert a 16×16 covariance matrix but it's rank 15 — mathematically singular, can't be inverted!Regularization (ledoit_wolf) ensures the covariance matrix is invertible and prevents this error.

##Stage 4-Classifier
Using balanced with the svm classifier gives higher weight to the target labels.
Conclusion:LDA performs poorly because EEG data violates its linearity assumption. SVM with RBF kernel captures non-linear patterns and handles class imbalance via class_weight='balanced', resulting in significantly better performance. Class weight ='balanced' penalizes the the target lables more as compared to non target ones.

	##EEGNet
	Step 5- class_weights = compute_class_weight(
        	'balanced',
       		classes=np.unique(y_all),
       		y=y_all). This tells the model to give higher weight to targets since they are outnumbered by the non target epochs. If we don't do this our model just gets biased towards 		predicting non targets and will get high accuracy by just predicting non target everytime.
        Step 6- val accuracy was jumping around wildly.SO I introduced the following changes-
		EarlyStopping automatically stops training when the model stops improving.
		ReduceLROnPlateau automatically reduces the learning rate when the model stops improving.
      		Think of the model learning like walking down a hill to find the lowest point:
		High learning rate = taking big steps
		Low learning rate  = taking small steps
		Big steps:   fast but might overshoot the bottom ❌
		Small steps: slow but lands precisely at the bottom ✅
		I changed initial lr from default to one-tenth of default.
				
		 
		MSE computes the average squared difference between predictions and targets, while MAE takes absolute differences.These excel in regression (predicting continuous values 		like signal amplitude) but fail in classification. Our one-hot labels and softmax outputs are probabilities summing to 1—MSE/MAE ignore this, leading to poor gradients and 		slow convergence. Categorical Crossentropy is best for multi class probabilistic outputs.
	
		During training, Keras keeps dropout active even on validation data in some versions — meaning the val accuracy shown during fit() might be slightly underestimated.
		model.evaluate() turns dropout off completely, giving a cleaner final number.
##Evaluation
ITR(Information Transfer Rate)-It measures how fast a BCI system can communicate, combining both speed and accuracy into one number.
				Intuitive understanding of each part:
				log2(N) — Maximum possible information. With 36 characters, each correct selection gives log2(36) = 5.17 bits of information.More choices = more information 				per selection
				P*log2(P) — Information from correct selections:
				Higher accuracy → more of the 5.17 bits actually get through. P=1.0 → full 5.17 bits;P=0.5 → only partial bits.
				(1-P)*log2((1-P)/(N-1)) — Penalty for errors:
				Wrong selections waste time and add confusion. Higher error rate → bigger penalty
				× 60/T — Convert to per minute: T = 2.1 seconds per character;60/2.1 = ~28 characters attempted per minute
				What ITR values mean in practice:
				 20 bits/min		Poor — barely usable
				20-50 bits/min		Acceptable for basic communication
				50-100 bits/min		Good BCI system
				100 bits/min		Excellent - fast communication
				So ITR rewards systems that are BOTH fast AND accurate — making it the perfect single metric for comparing BCI systems!
Why we didn't use k-fold cross validation for EEGNet?-First — what is K-Fold Cross Validation?
							The problem with a simple train/test split:
							Split 1: train on first 80%, test on last 20% → accuracy: 92%
							Split 2: train on last 80%, test on first 20% → accuracy: 78%
							Which is the real accuracy? We don't know!
							K-Fold solves this by testing on every part of the data:
							Your data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  (10 samples, k=5)
							Fold 1: train [3,4,5,6,7,8,9,10]  test [1,2]   → accuracy: 90%
							Fold 2: train [1,2,5,6,7,8,9,10]  test [3,4]   → accuracy: 85%
							Fold 3: train [1,2,3,4,7,8,9,10]  test [5,6]   → accuracy: 88%
							Fold 4: train [1,2,3,4,5,6,9,10]  test [7,8]   → accuracy: 92%
							Fold 5: train [1,2,3,4,5,6,7,8]   test [9,10]  → accuracy: 87%
							Final accuracy = average = (90+85+88+92+87)/5 = 88.4%
							Regular K-Fold has a problem with imbalanced data:
							Regular K-Fold might create:
							Fold 1 test set: 100 NonTarget, 5 Target   (20:1 ratio!) ← wrong
							Fold 2 test set: 80 NonTarget, 20 Target   (4:1 ratio!)  ← wrong
							Each fold has a different class ratio — results are inconsistent!
							Stratified K-Fold ensures every fold maintains the same ratio
							Why is Stratified K-Fold NOT suitable for EEGNet?— it's extremely slow:
							LDA/SVM k-fold: takes seconds
							EEGNet k-fold:  takes hours
							Fold 1: build EEGNet → train from scratch → test
							Fold 2: build EEGNet → train from scratch → test
							Fold 3: build EEGNet → train from scratch → test
							...
          						That's why for EEGNet we use simple train/test split.
  