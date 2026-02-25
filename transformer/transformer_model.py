"""
Transformer-based Q-Network for Trading.

Architecture:
    Input (batch, seq_len, features)
        → Linear projection to d_model
        → Sinusoidal Positional Encoding
        → N × Transformer Encoder Blocks (Multi-Head Self-Attention + FFN)
        → Global Average Pooling
        → Dense Head → Q-values (3 actions: Hold, Buy, Sell)

Uses native TF2/Keras layers (MultiHeadAttention, LayerNormalization).
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model


# ---------------------------------------------------------------------------
# Positional Encoding (sinusoidal – Vaswani et al. 2017)
# ---------------------------------------------------------------------------
class PositionalEncoding(layers.Layer):
    """Adds sinusoidal positional information to the input embeddings."""

    def __init__(self, max_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.d_model = d_model

        # Pre-compute the positional encoding matrix  (1, max_len, d_model)
        pe = np.zeros((max_len, d_model), dtype=np.float32)
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        self.pe = tf.constant(pe[np.newaxis, :, :])  # (1, max_len, d_model)

    def call(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len, "d_model": self.d_model})
        return config


# ---------------------------------------------------------------------------
# Transformer Encoder Block
# ---------------------------------------------------------------------------
class TransformerEncoderBlock(layers.Layer):
    """
    Single Transformer encoder block:
        LayerNorm → MultiHeadAttention → Dropout → Residual
        LayerNorm → Feed-Forward (expand → GELU → project) → Dropout → Residual
    Uses Pre-Norm layout (more stable for RL training).
    """

    def __init__(self, d_model: int, num_heads: int, dff: int,
                 dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.mha = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=d_model // num_heads,
            dropout=dropout_rate,
        )
        self.drop1 = layers.Dropout(dropout_rate)

        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = tf.keras.Sequential([
            layers.Dense(dff, activation="gelu"),
            layers.Dense(d_model),
        ])
        self.drop2 = layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        # Pre-Norm Self-Attention
        x_norm = self.ln1(x)
        attn_output = self.mha(x_norm, x_norm, training=training)
        attn_output = self.drop1(attn_output, training=training)
        x = x + attn_output  # Residual

        # Pre-Norm Feed-Forward
        x_norm = self.ln2(x)
        ffn_output = self.ffn(x_norm, training=training)
        ffn_output = self.drop2(ffn_output, training=training)
        x = x + ffn_output  # Residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "dropout_rate": self.dropout_rate,
        })
        return config


# ---------------------------------------------------------------------------
# Build full Transformer DQN model
# ---------------------------------------------------------------------------
def build_transformer_dqn(
    feature_dim: int,
    action_dim: int,
    seq_len: int = 64,
    d_model: int = 64,
    num_heads: int = 4,
    num_layers: int = 2,
    dff: int = 128,
    dropout_rate: float = 0.1,
) -> Model:
    """
    Builds a Transformer encoder that maps a sequence of market states
    to Q-values for each trading action.

    Args:
        feature_dim:  Number of features per timestep (e.g. 7).
        action_dim:   Number of discrete actions (3: Hold/Buy/Sell).
        seq_len:      Length of the input sequence window.
        d_model:      Internal dimension of the Transformer.
        num_heads:    Number of attention heads.
        num_layers:   Number of stacked encoder blocks.
        dff:          Hidden dimension of the feed-forward sub-network.
        dropout_rate: Dropout probability.

    Returns:
        A compiled Keras Model.
    """

    inputs = layers.Input(shape=(seq_len, feature_dim), name="state_sequence")

    # --- Project raw features to d_model dimension ---
    x = layers.Dense(d_model, name="input_projection")(inputs)

    # --- Positional Encoding ---
    x = PositionalEncoding(max_len=seq_len, d_model=d_model, name="pos_enc")(x)

    # --- Transformer Encoder Stack ---
    for i in range(num_layers):
        x = TransformerEncoderBlock(
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            dropout_rate=dropout_rate,
            name=f"encoder_block_{i}",
        )(x)

    # --- Final LayerNorm (common in Pre-Norm architectures) ---
    x = layers.LayerNormalization(epsilon=1e-6, name="final_ln")(x)

    # --- Global Average Pooling over the time axis ---
    x = layers.GlobalAveragePooling1D(name="global_pool")(x)

    # --- Q-value head ---
    x = layers.Dense(64, activation="relu", name="q_head_dense")(x)
    x = layers.Dropout(dropout_rate, name="q_head_dropout")(x)
    q_values = layers.Dense(action_dim, activation="linear", name="q_values")(x)

    model = Model(inputs=inputs, outputs=q_values, name="TransformerDQN")
    return model


# ---------------------------------------------------------------------------
# Quick sanity-check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_transformer_dqn(feature_dim=7, action_dim=3)
    model.summary()

    # Test forward pass
    dummy = tf.random.normal((2, 64, 7))
    out = model(dummy, training=False)
    print(f"\nForward pass OK — output shape: {out.shape}")  # (2, 3)
