import numpy as np

def extract_epochs_per_char(signal, flashing, stim_code, stim_type=None, window=240):
    all_segments, all_labels, all_codes = [], [], []
    n_epochs, n_samples, _ = signal.shape

    for epoch in range(n_epochs):
        segments, labels, codes = [], [], []
        for t in range(1, n_samples):
            if flashing[epoch, t] == 1 and flashing[epoch, t-1] == 0:
                code = int(stim_code[epoch, t])
                if code == 0 or t + window > n_samples:
                    continue
                segments.append(signal[epoch, t:t+window])
                codes.append(code)
                if stim_type is not None:
                    labels.append(int(stim_type[epoch, t]))

        all_segments.append(np.array(segments))
        all_codes.append(np.array(codes))

        if stim_type is not None:
            all_labels.append(np.array(labels))

    return all_segments, all_labels, all_codes