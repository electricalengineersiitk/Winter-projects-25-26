import numpy as np
from scipy.signal import butter,filtfilt,iirnotch

def bandpass_filter(signal,low=0.1,high=30.0,sfreq=240):
    b,a=butter(4,[low/(sfreq/2),high/(sfreq/2)],btype='band')
    return filtfilt(b,a,signal,axis=1)

def notch_filter(signal,freq=50.0,sfreq=240):
    b,a=iirnotch(freq/(sfreq/2),Q=30)
    return filtfilt(b,a,signal,axis=1)

def extract_epochs(signal,flashing,stimulus,stim_type=None,window=240):
    signal=bandpass_filter(signal)
    signal=notch_filter(signal)
    all_segments=[]
    all_labels=[]
    all_codes=[]
    n_chars,n_samples,_=signal.shape

    for i in range(n_chars):
        segs,labs,codes=[],[],[]

        for t in range(1,n_samples):
            if flashing[i,t]==1 and flashing[i,t-1]==0:
                code=int(stimulus[i,t])
                if code==0:
                    continue
                if t+window>n_samples:
                    continue
                segs.append(signal[i,t:t+window,:])
                codes.append(code)
                if stim_type is not None:
                    labs.append(int(stim_type[i,t]))

        all_segments.append(np.array(segs))
        all_codes.append(np.array(codes))

        if stim_type is not None:
            all_labels.append(np.array(labs))
            
    return all_segments,all_labels,all_codes