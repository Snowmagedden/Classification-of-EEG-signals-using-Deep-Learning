# Classification-of-EEG-signals-using-Deep-Learning
## 1 Task - Classification of EEG
Analyzing sequential data or time series is a very relevant and explored task in Deep Learning.
This kind of data appears in many domains and different formats; for example, stock prices,
videos and electrophysiological signals. In this task, you will be working with Electroencephalography (EEG) or brain waves. The
data set consists of EEG signals of different subjects while sleeping. There are different stages
of sleep characterized by specific kinds of EEG signals. Each sleep stage is then a period during
which the EEG signals present specific features or patterns, such as particular frequencies as
can be seen in 1 (The frequency bands in the table are averages, and other sources might define
them slightly differently, but these are representative) - For a more comprehensive description,
please go to https://www.ncbi.nlm.nih.gov/books/NBK10996/ You do not need to use all of
the information regarding the definition of stages unless you find it useful or want to go more
in-depth about the problem.
  The task you have to address is to classify EEG signals into sleep stages.

### 1.1 Data set
The data-set consists of EEG sequences of 3000-time steps each and coming from two electrode
locations on the head (Fpz-Cz and Pz-Oz) sampled at 100 Hz. That means that each sample contains
two signals of 3000 samples and that those samples correspond to 30 seconds of recording.
The labels that come along with the data specify six stages labelling them with corresponding
numbers as specified in table 1:

| label | Stage | Typical Frequencies (Hz) |
| :---:| :---: | :---: |
|0 |R |15-30|
|1 |1 |4-8|
|2 |2 |8-15|
|3 |3 |1-4|
|4 |4 |0.5-2|
|5 |W |15 - 50|

Table 1: Caption

W corresponds to the Wake stage, and R to REM sleep also called rapid eye movement, and
most commonly known as the dreaming stage.
Each sequence in the data set contains only one stage, which is specified by the corresponding
label.
Your final model should be evaluated on a given unlabeled Test set that will be provided
later.
The data set is presented in two different formats, Raw signals and Spectrograms; To solve
the problem, you can use either or both formats, there is no restriction.

### 1.1.1 Raw signals
The file 0Data_Raw_signals.pkl0 contains the sequences and the corresponding labels as two array
[sequences, labels].

### 1.1.2 Spectrograms
The file 0Data_Spectrograms.pkl0 contains the spectrograms of the sequences and the corresponding
labels as two array [spectrograms, labels]. A spectrogram is a visual representation of the spectrum of frequencies of a signal as it varies
with time. So a spectrogram is a 2D array, where one axis represents frequencies, and the other
represents time. https://en.wikipedia.org/wiki/Spectrogram
In the file 0Data_Spectrograms.pkl0 the spectrograms have a size 100 by 30 for each signal,
and they represent the same 3000-time steps EEG sequences as in the raw data. That is, for each
sequence in the raw data file, there is a corresponding spectrogram.
The spectrograms in 0Data_Spectrograms.pkl0 represent the frequencies of the signals in steps
of 0.5Hz between 0.5 and 50 Hz (Hence 100). Such frequencies correspond to the spectral information
in time windows of size 100-time steps each; thus, for a sequence of 3000-time steps, there
are 30 windows (hence spectrogram size: 100 by 30, frequencies by the number of windows).

