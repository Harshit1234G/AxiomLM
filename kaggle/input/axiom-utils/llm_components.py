import tensorflow as tf
import numpy as np
from sentencepiece import SentencePieceProcessor


# ----------------------------
# Hyperparameters
# ----------------------------
SEQUENCE_LEN = 256
BATCH_SIZE = 64
SHUFFLE_BUFFER = 10_000
N_EMBEDS = 512
N_HEADS = 8
N_BLOCKS = 6
N_EPOCHS = 8


# ----------------------------
# Tokenizer
# ----------------------------
def load_sp_tokenizer(path: str = 'sp_tokenizer.model') -> SentencePieceProcessor:
    sp = SentencePieceProcessor()
    sp.load(path)
    return sp

# ----------------------------
# Dataset pipeline
# ----------------------------
class LMDatasetLoader:
    def __init__(
        self,
        tokenizer: SentencePieceProcessor,
        *,
        seq_len: int = SEQUENCE_LEN,
        batch_size: int = BATCH_SIZE,
        shuffle_buffer: int = SHUFFLE_BUFFER
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def _sp_tokenize(self, line: tf.Tensor) -> tf.Tensor:
        tokens = self.tokenizer.encode(
            line.numpy().decode('utf-8'), 
            out_type= int
        )
        return tf.constant(tokens, dtype= tf.int32)

    def _tf_sp_tokenize(self, line: tf.Tensor) -> tf.Tensor:
        tokens = tf.py_function(
            func= self._sp_tokenize,
            inp= [line],
            Tout= tf.int32
        )
        tokens.set_shape([None])
        return tokens
    
    def create(self, text_file: str, training: bool) -> tf.data.Dataset:
        AUTOTUNE = tf.data.AUTOTUNE

        # Loading file
        ds = tf.data.TextLineDataset(text_file)

        # Tokenize
        ds = ds.map(
            self._tf_sp_tokenize,
            num_parallel_calls= AUTOTUNE
        )
        # Flatten into one continuous token stream
        ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)

        # create sliding windows of tokens
        ds = ds.window(
            self.seq_len + 1,
            shift= self.seq_len,
            drop_remainder= True,
        )

        ds = ds.flat_map(
            lambda w: w.batch(self.seq_len + 1, drop_remainder= True)
        )

        # Split into (input, target)
        ds = ds.map(
            lambda x: (x[:-1], x[1:]),
            num_parallel_calls= AUTOTUNE
        )

        if training:
            ds = ds.shuffle(self.shuffle_buffer)
            ds = ds.repeat()

        ds = ds.batch(self.batch_size, drop_remainder= True)
        ds = ds.prefetch(AUTOTUNE)
        
        return ds
    

# --------------------------------
# Sinusoidal Positional Encoding
# --------------------------------
@tf.keras.utils.register_keras_serializable()
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        max_seq_len: int = SEQUENCE_LEN,
        embed_size: int = N_EMBEDS,
        dtype: tf.dtypes.DType = tf.float32,
        **kwargs
    ) -> None:
        super().__init__(dtype= dtype, **kwargs)
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size
        assert self.embed_size % 2 == 0, 'embed_size must be even'

        # p -> each column is a position index
        # i -> each row corresponds to even embedding dimensions
        p, i = np.meshgrid(
            np.arange(self.max_seq_len),
            2 * np.arange(self.embed_size // 2)
        )
        # initializing PE matrix
        pos_emb = np.empty((1, self.max_seq_len, self.embed_size))   # (Batch, Positions, Embedding dimensions)
        pos_emb[0, :, ::2] = np.sin(p / 10_000 ** (i / self.embed_size)).T    # sine on even dimensions
        pos_emb[0, :, 1::2] = np.cos(p / 10_000 ** (i / self.embed_size)).T   # cosine on odd dimensions
        self.pos_encodings = tf.constant(pos_emb.astype(self.dtype))
        self.supports_masking = True    # propagates the input's automatic mask to next layer

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        batch_max_length = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :batch_max_length]
    
    def get_config(self) -> dict:
        base_config = super().get_config()
        return {
            **base_config,
            'max_seq_len': self.max_seq_len,
            'embed_size': self.embed_size
        }


# ----------------------------
# Layer Normalization
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, *, epsilon: float = 1e-3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape: tuple[int, ...]) -> None:
        self.gamma = self.add_weight(
            name= 'gamma', 
            shape= input_shape[-1:],
            initializer= 'ones',
            trainable= True
        )
        
        self.beta = self.add_weight(
            name= 'beta', 
            shape= input_shape[-1:],
            initializer= 'zeros',
            trainable= True
        )

    def call(self, X: tf.Tensor) -> tf.Tensor:
        mean, variance = tf.nn.moments(X, axes= -1, keepdims= True)
        return self.gamma * (X - mean) / (tf.sqrt(variance + self.epsilon)) + self.beta

    def get_config(self) -> dict:
        base_config = super().get_config()
        return {
            **base_config,
            'epsilon': self.epsilon
        }


# ----------------------------
# Attention Mechanism
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class MultiHeadedAttention(tf.keras.layers.Layer):
    def __init__(
        self, 
        n_embeds: int, 
        n_heads: int, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        assert n_embeds % n_heads == 0

        self.n_heads = n_heads
        self.head_dim = n_embeds // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(
            3 * n_embeds,
            use_bias= False
        )
        self.proj = tf.keras.layers.Dense(n_embeds)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        qkv = self.qkv(x)  # (B, T, 3 * C)
        q, k, v = tf.split(qkv, 3, axis= -1)

        q = tf.reshape(q, (B, T, self.n_heads, self.head_dim))
        k = tf.reshape(k, (B, T, self.n_heads, self.head_dim))
        v = tf.reshape(v, (B, T, self.n_heads, self.head_dim))

        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, T, D)
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        att = tf.matmul(q, k, transpose_b= True) * self.scale

        mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
        mask = tf.reshape(mask, (1, 1, T, T))
        att = tf.where(mask == 0, -1e9, att)

        att = tf.nn.softmax(att, axis= -1)

        out = tf.matmul(att, v)  # (B, H, T, D)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (B, T, -1))

        return self.proj(out)


@tf.keras.utils.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self, 
        n_embed: int,
        *,
        activation: str = 'gelu',
        kernel_initializer: str = 'he_normal',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.activation = tf.keras.activations.get(activation)

        self.net = tf.keras.Sequential([
            tf.keras.layers.Dense(
                4 * n_embed, 
                activation= self.activation,
                kernel_initializer= kernel_initializer
            ),
            tf.keras.layers.Dense(n_embed)
        ])

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.net(x)
    

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        n_embeds: int = N_EMBEDS,
        n_heads: int = N_HEADS,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.n_embeds = n_embeds
        self.n_heads = n_heads

        self.ln1 = LayerNormalization()
        self.ln2 = LayerNormalization()

        self.attn = MultiHeadedAttention(
            n_embeds= self.n_embeds,
            n_heads= self.n_heads
        )

        self.ffwd = FeedForward(n_embed= self.n_embeds)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = x + self.attn(self.ln1(x), training= training)
        x = x + self.ffwd(self.ln2(x), training= training)
        return x

    def get_config(self) -> dict:
        base_config = super().get_config()
        return {
            **base_config,
            'n_embeds': self.n_embeds,
            'n_heads': self.n_heads
        }


# ----------------------------
# Metric
# ----------------------------
@tf.keras.utils.register_keras_serializable()
def perplexity(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits= True
    )

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask

    mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return tf.exp(mean_loss)
