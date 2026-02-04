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
        shuffle_buffer: int | None = None
    ) -> None:
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def sp_tokenize(self, line: tf.Tensor) -> tf.Tensor:
        tokens = self.tokenizer.encode(line.numpy().decode('utf-8'), out_type= int)
        return tf.constant(tokens, dtype= tf.int32)

    def tf_sp_tokenize(self, line: tf.Tensor) -> tf.Tensor:
        tokens = tf.py_function(
            func= self.sp_tokenize,
            inp= [line],
            Tout= tf.int32
        )
        tokens.set_shape([None])
        return tokens
    
    def create(self, text_file: str) -> tf.data.Dataset:
        AUTOTUNE = tf.data.AUTOTUNE

        # Loading file
        ds = tf.data.TextLineDataset(text_file)

        # Tokenize
        ds = ds.map(
            self.tf_sp_tokenize,
            num_parallel_calls= AUTOTUNE
        )
        # Flatten into one continuous token stream
        ds = ds.flat_map(
            lambda x: tf.data.Dataset.from_tensor_slices(x)
        )
        # creating sequences of length (seq_len + 1)
        ds = ds.batch(self.seq_len + 1, drop_remainder= True)

        # Split into (input, target)
        ds = ds.map(
            lambda x: (x[:-1], x[1:]),
            num_parallel_calls= AUTOTUNE
        )

        if self.shuffle_buffer is not None:
            ds = ds.shuffle(self.shuffle_buffer)

        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    

# --------------------------------
# Sinusoidal Positional Encoding
# --------------------------------
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
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon: float = 1e-3, **kwargs) -> None:
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


def softmax_with_temperature(logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
    logits = logits / temperature
    return tf.nn.softmax(logits)


# ----------------------------
# Attention Mechanism
# ----------------------------
