# Import necessary libraries
import librosa
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns

# READ FILE: Load the audio file
audio_path = 'audio_files/combined_0257_10s.wav'
y, sr = librosa.load(audio_path, sr=44100)

# STFT: Short Time Fourier Transform to convert time domain signal to frequency domain
n_fft = 512
hop_length = n_fft // 2
stft_result = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
# librosa.stft performs the STFT and returns complex-valued matrix of shape (freq_bins，time_steps)
# The shape of stft_result is (257, m) where 257 is the number of time bins and m is the number of frequency bins

# PSD: Power Spectral Density calculation from the STFT result
psd = np.abs(stft_result)**2
psd_db = 10 * np.log10(psd + np.finfo(float).eps)

# FCM
correlation_matrix = np.corrcoef(psd_db)

# Flip the correlation matrix upside down for better visualization
fc_matrix_flipped = np.flipud(correlation_matrix)


plt.figure(figsize=(10, 8))
# Create a heatmap using seaborn for visualizing the correlation matrix
ax = sns.heatmap(fc_matrix_flipped, cmap='coolwarm', square=True, xticklabels='auto', yticklabels='auto')

y_tick_labels = ax.get_yticklabels()
y_tick_labels = y_tick_labels[::-1]
ax.set_yticklabels(y_tick_labels)

# Show the plot
plt.title('Frequency Correlation Matrix')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Frequency(Hz)')
plt.show()
