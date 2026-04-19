import mne
import matplotlib.pyplot as plt
mne.viz.set_browser_backend('matplotlib')

def preprocessing (eeg_data):

  # Making RawObject in mne
  
  eeg_df = eeg_data.T
  sfreq = 256.0
  ch_names = ['Fz','Cz','Pz','Oz','P3','P4','PO7','PO8','F3','F4','FCz','C3','C4','CP3','CPz','CP4']

  info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
  raw = mne.io.RawArray(eeg_df, info)

  # Passing band_pass filter

  band_pass_filtered_raw = raw.copy()
  band_pass_filtered_raw.filter(l_freq=0.1, h_freq=30,fir_design='firwin')

  # Passing notch filter

  final_filtered_raw = band_pass_filtered_raw.copy()
  final_filtered_raw.notch_filter(freqs=50)

  # Setting montage

  montage = mne.channels.make_standard_montage('standard_1020')

  final_raw = final_filtered_raw.copy()

  final_raw.set_montage(montage)

  return final_raw




def ica_analysis(final_raw):

  ica = mne.preprocessing.ICA(n_components=.99 ,max_iter="auto", random_state=55)

  raw_for_ica = final_raw.copy().filter(l_freq=1., h_freq=None, fir_design='firwin', verbose=False)

  ica.fit(raw_for_ica)

  ica.plot_components()

  ica.plot_sources(raw_for_ica, start=0, stop=196)

  while True:
    n = int(input("Enter the component number to examine (or -1 to stop): "))
    if n == -1:
        break
    ica.plot_overlay(raw_for_ica, exclude=[n], picks="eeg")
    ica.plot_properties(raw_for_ica,picks = [n] ,reject=None)
    continue

  x = [int(x) for x in input("Enter the component numbers to exclude (or -1 if no components): ").split()]
  if x == [-1]:
    return final_raw
  
  artifact_components = x
  ica.exclude = artifact_components

  reconstructed_raw = final_raw.copy()
  ica.apply(reconstructed_raw)

  return reconstructed_raw




def bad_channel_rej (reconstructed_raw,threshold=1.2 ,n_neighbors=5):

  bad_channels = mne.preprocessing.find_bad_channels_lof(reconstructed_raw, n_neighbors=n_neighbors, threshold=threshold, return_scores=True)
  print(bad_channels)

  reconstructed_raw.info['bads'] = bad_channels[0]

  reconstructed_raw.set_eeg_reference(ref_channels='average')

  return reconstructed_raw





def make_epochs (event_arr,reconstructed_raw):

  # Create a mapping for event IDs to labels
  event_desc = {1:"Non-Target",2:"Target"}
  # Convert array to Annotations
  annot = mne.annotations_from_events(event_arr, sfreq=reconstructed_raw.info['sfreq'], event_desc=event_desc)
  # Add to raw object
  reconstructed_raw.set_annotations(annot)

  events, event__id = mne.events_from_annotations(reconstructed_raw)

  picks = mne.pick_types(reconstructed_raw.info, eeg=True, exclude='bads')

  epochs = mne.Epochs(reconstructed_raw, events, event_id=event__id, picks=picks, tmin=-0.2, tmax=0.8, baseline=None, preload=True)

  evoc_target = epochs['Target'].average()
  evoc_nontarget = epochs['Non-Target'].average()

  # Calculate difference
  evoc_diff = mne.EvokedArray(evoc_target.data - evoc_nontarget.data, evoc_target.info, tmin=evoc_target.times[0])

  # Use MNE's built-in vlines parameter for the most reliable marking
  fig_10 = mne.viz.plot_compare_evokeds(
      {'Target': evoc_target, 'Non-Target': evoc_nontarget, 'Difference': evoc_diff},
      title='P300 Response (Target vs Non-Target)',
      show_sensors=False,
      vlines=[0.3]  # This tells MNE specifically to mark 0.3s
  )

  # Secondary high-visibility manual check
  if fig_10 and len(fig_10) > 0:
      ax = fig_10[0].axes[0]
      print(f"Verified X-axis limits: {ax.get_xlim()}")
      # Drawing a very thick magenta dashed line specifically at x=0.3
      ax.axvline(x=0.3, color='magenta', linestyle='--', linewidth=5, alpha=0.8, zorder=100, label='Manual 0.3s Mark')
      ax.legend(loc='upper right')

  plt.show()

  return epochs