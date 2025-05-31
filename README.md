# 🌿🔬 Simulated Biosensor Network for Organophosphate Detection Using Information-Theoretic Optimization 📊✨

## Project Overview

This project delves into the fascinating world of **simulated biosensor networks** for detecting harmful **organophosphates**. We're simulating how plants might respond electrophysiologically to different levels of these contaminants, then using powerful machine learning techniques—from classic algorithms to advanced Convolutional Neural Networks (CNNs)—to classify these responses. A central theme is **information-theoretic optimization**, where we particularly lean on **Mutual Information** to gauge how well our models and features capture vital information about the contamination.

Our comprehensive pipeline covers:
* **🌱 Signal Simulation**: Crafting realistic time-series data mirroring plant reactions.
* **🧠 Feature Engineering**: Extracting crucial statistical and frequency-domain characteristics from the raw signals.
* **📈 Data Visualization**: Offering clear insights into our data and features through various plots (e.g., signal examples, feature correlations, dimensionality reduction).
* **🤖 Machine Learning Model Training**: Building and training both traditional classifiers (Random Forest, SVM, Gradient Boosting) and a CNN to detect organophosphate presence.
* **🎯 Performance Evaluation**: Assessing model effectiveness using metrics like accuracy, ROC AUC, and, critically, **Mutual Information**, highlighting our information-theoretic approach.

---

## Data Structure

Our simulated dataset comprises **electrophysiological time-series signals**, each 256 points long. These signals are designed to reflect plant responses to varying organophosphate exposure. Each signal comes with a **label** indicating its contamination level:
* `0`: Absolutely clean (no contamination) 💧
* `1-3`: Increasing levels of contamination ⬆️

To mimic real-world imperfections, we've introduced a `label_noise_ratio` of 0.05, meaning about 5% of labels are intentionally randomized. For our classification tasks, we simplify the problem into a **binary classification**: `0` for clean signals and `1` for any level of contamination.

---

## Getting Started

Ready to dive in? Here’s how to get this project up and running on your local machine for development and testing.

### Prerequisites

You'll need **Python 3.8+** installed. The project relies on several popular scientific computing and machine learning libraries.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
    cd your-repository-name
    ```

    *(**Note**: Remember to replace `your-username/your-repository-name.git` with your actual repository URL!)*

2.  **Create a virtual environment (highly recommended for a clean setup!):**

    ```bash
    python -m venv venv
    ```

    * **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    * **On Windows:**
        ```bash
        venv\Scripts\activate
        ```

3.  **Install the required packages:**

    ```bash
    pip install numpy matplotlib seaborn pandas tqdm scipy scikit-learn tensorflow joblib
    ```

### Running the Project

The entire analytical pipeline, from data generation to model evaluation, can be executed with a single command using the provided `setup_and_run.sh` script.

1.  **Make the script executable:**

    ```bash
    chmod +x setup_and_run.sh
    ```

2.  **Run the script:**

    ```bash
    ./setup_and_run.sh
    ```

This script will seamlessly handle everything:
* 📁 Setting up necessary directories (`biosensor_plots` for all your visualizations and `biosensor_data` for saved models and processed data).
* 🌊 Generating the simulated biosensor dataset.
* 📊 Extracting meaningful features from the raw signals.
* 🚀 Training both the advanced CNN and our classical machine learning models.
* 🖼️ Generating a suite of insightful plots (like confusion matrices, ROC curves, and feature correlation heatmaps).
* 💾 Saving all processed data, trained models, and performance results for your review.

---

## Code Structure

* `main.py`: (This is where the core Python logic provided would typically reside, or it could be modularized into several files for larger projects.) Contains the heart of our project: data simulation, feature extraction, model training, and rigorous evaluation.
* `setup_and_run.sh`: The handy shell script that orchestrates the execution of the Python script and manages directory creation.
* `biosensor_plots/`: Your gallery for all generated plots (saved as crisp PNG images).
* `biosensor_data/`: The treasure trove for processed data (NPY files), your trained models (Keras and PKL files), and the detailed performance results (CSV file).

---

## Key Components and Methodologies

### 🌿 Organophosphate Signal Simulator

Our `OrganophosphateSignalSimulator` class is the engine behind generating synthetic electrophysiological time-series signals. Each signal, a 256-point waveform, has its characteristics (like base frequency and decay) cleverly influenced by the contamination level. We've also sprinkled in some Gaussian noise and a `label_noise_ratio` to give our simulated data that authentic, real-world feel.

### 🧠 Feature Extraction

The `extract_features` function is where we transform raw signals into a rich set of descriptive features. These include:
* **Statistical features**: Mean, standard deviation, skewness, kurtosis, min, max, median, and signal power.
* **Frequency-domain features**: Mean, standard deviation, max, and min of the Fast Fourier Transform (FFT) magnitudes.

These features are carefully chosen to capture unique signal characteristics that directly correlate with contamination levels.

### 📈 Data Visualization

We've integrated several powerful visualization functions to help you deeply understand your data and the effectiveness of our feature engineering:
* `plot_sample_signals`: See a few raw signals up close, complete with their labels.
* `plot_label_distribution`: Get a quick overview of how samples are distributed across contamination levels.
* `plot_feature_correlation`: A heatmap revealing how different features relate to each other—great for spotting redundancy!
* `plot_3d_embedding`: Uses Principal Component Analysis (PCA) to project your features into 3D space, giving you a sense of how well classes might separate.
* `plot_tsne`: Employs t-Distributed Stochastic Neighbor Embedding (t-SNE) for a 2D projection that can uncover hidden clusters and class separation.

### 🤖 Machine Learning Models

We put several classification models to the test:
* **Convolutional Neural Network (CNN)**: A powerful `Sequential` Keras model, featuring `Conv1D` layers, `MaxPooling1D`, `Flatten`, `Dense` layers, and `Dropout`. This model learns directly from the raw, scaled time-series data.
* **Random Forest Classifier**: An ensemble powerhouse based on decision trees, trained on our meticulously extracted features.
* **Support Vector Machine (SVM)**: A robust classifier that intelligently finds the optimal hyperplane to distinguish between classes, also trained on extracted features.
* **Gradient Boosting Classifier**: Another ensemble gem that builds trees sequentially, continually refining its predictions.
* **Voting Classifier (Ensemble)**: The ultimate team player, combining the strengths of Random Forest, SVM, and Gradient Boosting to achieve potentially superior overall performance.

### 📊 Information-Theoretic Optimization and Evaluation

Beyond standard metrics, we put a special emphasis on **Information-Theoretic Optimization** by including **Mutual Information** as a crucial evaluation metric.

* **Accuracy**: The straightforward percentage of correct classifications.
* **ROC AUC (Receiver Operating Characteristic Area Under the Curve)**: A robust measure of a classifier's ability to discriminate between classes. Higher AUC values mean better performance. We also plot the full ROC curves for visual analysis.
* **Mutual Information**: This metric quantifies the **amount of information** one variable provides about another. In our case, it measures how much information the model's predictions give us about the true contamination levels. A higher mutual information score directly indicates that our model's output is more informative and strongly related to the actual organophosphate status. This is central to our "information-theoretic optimization" approach!

---

## Results

You'll find a detailed summary of each model's performance (CNN, Random Forest, SVM, Gradient Boosting, and Ensemble) in `model_results.csv` and also printed directly to your console. This table includes **Accuracy**, **Mutual Information**, and **AUC scores** for each model, giving you a holistic view.

Furthermore, we generate a variety of helpful plots, all saved in the `biosensor_plots` directory:
* **Confusion Matrices**: For every trained model, clearly showing true vs. predicted counts for our binary classification.
* **ROC Curves**: A visual representation of each model's true positive rate against its false positive rate, illustrating its discriminative power.

---

## Future Work

The journey doesn't stop here! There are many exciting avenues for expanding this project:

* **🧪 Advanced Signal Processing**: Let's explore more sophisticated techniques like wavelet transforms or empirical mode decomposition for even richer feature extraction.
* **🧠 Deeper Learning Architectures**: Experiment with more intricate CNN designs, Recurrent Neural Networks (LSTMs), or even Transformer models for time-series classification.
* **🔍 Unsupervised Insights**: Can clustering techniques on raw signals or features reveal natural groupings related to contamination without explicit labels?
* **⚙️ Hyperparameter Nirvana**: Implement systematic hyperparameter tuning to squeeze every last drop of performance from our models.
* **🌍 Real-world Validation**: The ultimate test! Integrate and validate our framework with actual biosensor data from real organophosphate exposure scenarios.
* **🔢 Multi-class Mastery**: Instead of binary classification, let's directly predict all four distinct contamination levels (0-3).
* **🌐 Sensor Network Simulation**: Introduce concepts of a distributed sensor network, including advanced data fusion strategies from multiple simulated biosensors.
* **🗺️ Active Learning & Optimal Sensor Placement**: Investigate how information-theoretic approaches can guide optimal sensor placement or active sampling strategies within a network to maximize information gain with minimal resources.

---

Feel free to contribute, explore, and enhance this project! If you have any questions or ideas, don't hesitate to open an issue or reach out.
