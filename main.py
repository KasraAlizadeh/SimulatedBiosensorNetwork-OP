# === Import Required Libraries === #
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
from tqdm import tqdm
from scipy.fft import fft
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mutual_info_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib
import os
import joblib

# === Set Global Configurations === #
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['font.family'] = 'Arial'

# === Ensure Reproducibility === #
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# === Directory Setup for Saving Outputs === #
plot_dir = "biosensor_plots"
data_dir = "biosensor_data"
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# === Organophosphate Signal Simulator === #
class OrganophosphateSignalSimulator:
    def __init__(self, n_samples=10000, signal_length=256, label_noise_ratio=0.05):
        self.n_samples = n_samples
        self.signal_length = signal_length
        self.label_noise_ratio = label_noise_ratio

    def generate_waveform(self, level):
        t = np.linspace(0, 1, self.signal_length)
        base_freq = 5 + level
        decay = np.exp(-level * t)
        noise = np.random.normal(0, 0.2, size=self.signal_length)
        waveform = np.sin(2 * np.pi * base_freq * t) * decay + noise
        return waveform

    def generate_dataset(self):
        levels = np.random.randint(0, 4, size=self.n_samples)
        data = np.array([self.generate_waveform(lvl) for lvl in levels])
        noisy_labels = levels.copy()
        n_noise = int(self.label_noise_ratio * self.n_samples)
        idx = np.random.choice(self.n_samples, n_noise, replace=False)
        noisy_labels[idx] = np.random.randint(0, 4, size=n_noise)
        return data, noisy_labels, levels

# === Feature Extraction === #
def extract_features(data):
    features = []
    for signal in data:
        f = np.abs(fft(signal))[:len(signal)//2]
        stats = [
            np.mean(signal), np.std(signal), skew(signal), kurtosis(signal),
            np.max(signal), np.min(signal), np.median(signal),
            np.sum(np.square(signal)) / len(signal)
        ]
        freq_stats = [
            np.mean(f), np.std(f), np.max(f), np.min(f)
        ]
        features.append(stats + freq_stats)
    return np.array(features)

# === Visualization Functions === #
def plot_sample_signals(data, labels, save_path=None, n=5):
    plt.figure(figsize=(10, 6))
    for i in range(n):
        plt.plot(data[i], label=f"Label: {labels[i]}")
    plt.title("Sample Simulated Signals")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

def plot_dataset_summary(data, labels):
    avg_signal = np.mean(data, axis=0)
    plt.figure(figsize=(8, 4))
    plt.plot(avg_signal)
    plt.title("Average Signal Across Dataset")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

def plot_feature_correlation(features, save_path=None):
    corr = np.corrcoef(features.T)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', square=True)
    plt.title("Feature Correlation Matrix")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

def plot_3d_embedding(features, labels, save_path=None):
    pca = PCA(n_components=3)
    components = pca.fit_transform(features)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    p = ax.scatter(components[:, 0], components[:, 1], components[:, 2], c=labels, cmap='viridis', s=10)
    fig.colorbar(p)
    ax.set_title("3D PCA Embedding")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

def plot_model_roc(y_true, y_prob, model_name, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

# === Additional Helper Functions === #
def save_data(features, labels, prefix="binary"):
    np.save(os.path.join(data_dir, f"X_{prefix}.npy"), features)
    np.save(os.path.join(data_dir, f"y_{prefix}.npy"), labels)
    print(f"Saved dataset: X_{prefix}.npy, y_{prefix}.npy")

def plot_label_distribution(labels, save_path=None):
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts, hue=unique, palette="viridis", legend=False)
    plt.xlabel("Contamination Level")
    plt.ylabel("Count")
    plt.title("Label Distribution in Dataset")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

def plot_tsne(features, labels, save_path=None):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    X_tsne = tsne.fit_transform(features)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', s=10)
    plt.legend(*scatter.legend_elements(), title="Labels")
    plt.title("2D t-SNE Projection of Feature Space")
    if save_path:
        plt.savefig(save_path)
    plt.tight_layout()
    plt.show()

# === Main Execution Pipeline === #
print("[1] Generating dataset...")
simulator = OrganophosphateSignalSimulator(n_samples=25000, label_noise_ratio=0.05)
X_raw, y_multi, levels = simulator.generate_dataset()
plot_sample_signals(X_raw, y_multi, save_path=os.path.join(plot_dir, "sample_signals.png"))
plot_label_distribution(y_multi, save_path=os.path.join(plot_dir, "label_distribution.png"))
plot_dataset_summary(X_raw, y_multi)

y_binary = np.where(y_multi == 0, 0, 1)
save_data(X_raw, y_binary, prefix="raw")

print("[2] Extracting features...")
X_feat = extract_features(X_raw)
plot_feature_correlation(X_feat, save_path=os.path.join(plot_dir, "feature_correlation.png"))
plot_3d_embedding(X_feat, y_binary, save_path=os.path.join(plot_dir, "3d_embedding.png"))
plot_tsne(X_feat, y_binary, save_path=os.path.join(plot_dir, "tsne_projection.png"))
save_data(X_feat, y_binary, prefix="features")

print("[3] Preparing data...")
X_train_f, X_test_f, y_train, y_test = train_test_split(X_feat, y_binary, test_size=0.2, random_state=42)
X_train_r, X_test_r, _, _ = train_test_split(X_raw, y_binary, test_size=0.2, random_state=42)

scaler_f = StandardScaler()
X_train_f = scaler_f.fit_transform(X_train_f)
X_test_f = scaler_f.transform(X_test_f)
joblib.dump(scaler_f, os.path.join(data_dir, "scaler_features.pkl"))

scaler_r = StandardScaler()
X_train_r = scaler_r.fit_transform(X_train_r)
X_test_r = scaler_r.transform(X_test_r)
X_train_r = X_train_r[..., np.newaxis]
X_test_r = X_test_r[..., np.newaxis]
joblib.dump(scaler_r, os.path.join(data_dir, "scaler_raw.pkl"))

print("[4] Training CNN model...")
cnn = Sequential([
    Input(shape=(256, 1)),
    Conv1D(32, 3, activation='relu'),
    MaxPooling1D(2),
    Conv1D(64, 3, activation='relu'),
    MaxPooling1D(2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
cnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train_r, y_train, epochs=25, batch_size=64, verbose=1)
cnn.save(os.path.join(data_dir, "cnn_model.keras"))

y_cnn_proba = cnn.predict(X_test_r).flatten()
y_cnn = (y_cnn_proba > 0.5).astype(int)

fig, ax = plt.subplots()
ConfusionMatrixDisplay(confusion_matrix(y_test, y_cnn)).plot(ax=ax)
ax.set_title("Confusion Matrix - CNN")
fig.tight_layout()
fig.savefig(os.path.join(plot_dir, "confusion_matrix_cnn.png"))
plt.show()

print("[5] Training classical ML models...")
rf = RandomForestClassifier(n_estimators=100)
svm = SVC(probability=True)
gbm = GradientBoostingClassifier()
ensemble = VotingClassifier(estimators=[('rf', rf), ('svm', svm), ('gbm', gbm)], voting='soft')

models = {'CNN': (y_cnn, y_cnn_proba)}
plot_model_roc(y_test, y_cnn_proba, 'CNN', save_path=os.path.join(plot_dir, "roc_cnn.png"))

for name, model in zip(['Random Forest', 'SVM', 'Gradient Boosting', 'Ensemble'], [rf, svm, gbm, ensemble]):
    model.fit(X_train_f, y_train)
    joblib.dump(model, os.path.join(data_dir, f"{name.lower().replace(' ', '_')}_model.pkl"))
    y_pred = model.predict(X_test_f)
    y_prob = model.predict_proba(X_test_f)[:, 1]
    models[name] = (y_pred, y_prob)
    fig, ax = plt.subplots()
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot(ax=ax)
    ax.set_title(f"Confusion Matrix - {name}")
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"confusion_matrix_{name.lower().replace(' ', '_')}.png"))
    plt.show()
    plot_model_roc(y_test, y_prob, name, save_path=os.path.join(plot_dir, f"roc_{name.lower().replace(' ', '_')}.png"))

# === Evaluation Summary Table === #
data = []
for name, (yhat, yprob) in models.items():
    acc = accuracy_score(y_test, yhat)
    mi = mutual_info_score(y_test, yhat)
    auc_score = roc_auc_score(y_test, yprob)
    data.append([name, acc, mi, auc_score])
df_results = pd.DataFrame(data, columns=['Model', 'Accuracy', 'Mutual Info', 'AUC'])
df_results.to_csv(os.path.join(data_dir, "model_results.csv"), index=False)
print("\nModel Performance Summary:")
print(df_results)

