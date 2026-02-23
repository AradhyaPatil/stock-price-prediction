"""
model.py — Advanced LSTM model for stock price prediction.
Bidirectional LSTM with Attention mechanism using Keras/TensorFlow.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout,
    LayerNormalization, Layer, Multiply, Permute,
    RepeatVector, Flatten, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ──────────────────────────────────────────────────────────────
# Attention Layer
# ──────────────────────────────────────────────────────────────

class AttentionLayer(Layer):
    """
    Soft attention layer — learns which time-steps in the sequence
    are most important for the prediction.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            name="attention_context",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, x):
        # x shape: (batch, time_steps, features)
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)  # (batch, T, F)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)  # (batch, T, 1)
        context = tf.reduce_sum(x * attention_weights, axis=1)  # (batch, F)
        return context

    def get_config(self):
        return super().get_config()


# ──────────────────────────────────────────────────────────────
# Model Builder
# ──────────────────────────────────────────────────────────────

def build_lstm_model(input_shape: tuple, units: int = 128):
    """
    Build an advanced Bidirectional LSTM with Attention.

    Architecture:
        Input → BiLSTM(128) → LayerNorm → Dropout(0.15)
              → BiLSTM(64)  → LayerNorm → Dropout(0.15)
              → BiLSTM(32)  → LayerNorm → Dropout(0.1)
              → Attention
              → Dense(64, ReLU) → Dropout(0.1)
              → Dense(32, ReLU) → Dense(1)

    Args:
        input_shape: (time_steps, num_features)
        units:       Base number of LSTM units

    Returns:
        Compiled Keras Model.
    """
    inputs = Input(shape=input_shape)

    # First BiLSTM layer
    x = Bidirectional(LSTM(units, return_sequences=True))(inputs)
    x = LayerNormalization()(x)
    x = Dropout(0.15)(x)

    # Second BiLSTM layer
    x = Bidirectional(LSTM(units // 2, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.15)(x)

    # Third BiLSTM layer
    x = Bidirectional(LSTM(units // 4, return_sequences=True))(x)
    x = LayerNormalization()(x)
    x = Dropout(0.1)(x)

    # Attention mechanism
    x = AttentionLayer()(x)

    # Dense head
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1)(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss="mse",
        metrics=["mae"]
    )

    return model


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

class WarmUpScheduler(tf.keras.callbacks.Callback):
    """Warm up learning rate during first few epochs."""
    def __init__(self, target_lr=0.0005, warmup_epochs=5):
        super().__init__()
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.target_lr * (epoch + 1) / self.warmup_epochs
            try:
                self.model.optimizer.learning_rate.assign(lr)
            except Exception:
                try:
                    tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                except Exception:
                    pass


def train_model(model, X_train, y_train, epochs: int = 50, batch_size: int = 32,
                validation_split: float = 0.1, use_early_stopping: bool = True):
    """
    Train the LSTM model with EarlyStopping, LR scheduling, and warmup.
    """
    callbacks = []

    # Warm up LR for first 5 epochs
    callbacks.append(WarmUpScheduler(target_lr=0.0005, warmup_epochs=5))

    if use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=20,
                restore_best_weights=True,
                verbose=0
            )
        )

    # Reduce LR when validation loss plateaus
    callbacks.append(
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=8,
            min_lr=1e-6,
            verbose=0
        )
    )

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=callbacks,
        shuffle=True,
        verbose=0
    )

    return history


# ──────────────────────────────────────────────────────────────
# Prediction
# ──────────────────────────────────────────────────────────────

def predict(model, X) -> np.ndarray:
    """Run predictions on input data."""
    return model.predict(X, verbose=0).flatten()


def predict_future(model, last_sequence: np.ndarray, scaler, days: int = 30,
                   close_idx: int = 0, num_features: int = 1,
                   n_simulations: int = 50) -> dict:
    """
    Fast future prediction: single model run + percentage-based volatility.

    1. Runs the model ONCE to get the deterministic trend.
    2. Converts historical data to real prices to compute daily return volatility.
    3. Overlays a random walk using that volatility for realistic oscillation.

    Returns:
        dict with 'median', 'mean', 'upper', 'lower', 'paths'
    """
    # ── Step 1: Single deterministic model run ──
    trend_scaled = []
    current_seq = last_sequence.copy()

    for _ in range(days):
        input_seq = current_seq.reshape(1, current_seq.shape[0], current_seq.shape[1])
        pred = model.predict(input_seq, verbose=0)[0, 0]
        trend_scaled.append(pred)

        new_row = current_seq[-1].copy()
        new_row[close_idx] = pred
        current_seq = np.vstack([current_seq[1:], new_row.reshape(1, -1)])

    trend_scaled = np.array(trend_scaled)

    # Convert trend to real prices
    dummy = np.zeros((days, num_features))
    dummy[:, close_idx] = trend_scaled
    trend_real = scaler.inverse_transform(dummy)[:, close_idx]

    # ── Step 2: Compute daily return volatility from REAL historical prices ──
    # Convert the historical close sequence to real prices
    hist_dummy = np.zeros((len(last_sequence), num_features))
    hist_dummy[:, close_idx] = last_sequence[:, close_idx]
    hist_real = scaler.inverse_transform(hist_dummy)[:, close_idx]

    # Daily percentage returns
    daily_returns = np.diff(hist_real) / hist_real[:-1]
    daily_vol_pct = np.std(daily_returns) if len(daily_returns) > 1 else 0.01

    # ── Step 3: Generate paths using percentage-based random walk ──
    all_paths = np.zeros((n_simulations, days))
    for sim in range(n_simulations):
        noise_pct = np.zeros(days)
        for t in range(days):
            # Slightly increasing uncertainty further out
            horizon_factor = 1.0 + 0.2 * (t / max(days, 1))
            noise_pct[t] = (noise_pct[t - 1] if t > 0 else 0)
            noise_pct[t] += np.random.normal(0, daily_vol_pct * horizon_factor)
        # Apply cumulative percentage noise to the trend
        all_paths[sim] = trend_real * (1.0 + noise_pct)

    # Pick the path closest to the overall median as the display line
    median_vals = np.median(all_paths, axis=0)
    endpoint_diffs = np.abs(all_paths[:, -1] - median_vals[-1])
    best_path_idx = np.argmin(endpoint_diffs)

    return {
        "median": all_paths[best_path_idx],
        "mean": np.mean(all_paths, axis=0),
        "upper": np.percentile(all_paths, 90, axis=0),
        "lower": np.percentile(all_paths, 10, axis=0),
        "paths": all_paths,
    }
