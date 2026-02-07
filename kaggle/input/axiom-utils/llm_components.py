import tensorflow as tf
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
def load_sp_tokenizer(path: str) -> SentencePieceProcessor:
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
        shift: int,
        *,
        seq_len: int = SEQUENCE_LEN,
        batch_size: int = BATCH_SIZE,
        shuffle_buffer: int = SHUFFLE_BUFFER
    ) -> None:
        self.tokenizer = tokenizer
        self.shift = shift
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def _sp_tokenize(self, line):
        tokens = self.tokenizer.encode(
            line.numpy().decode('utf-8'), 
            out_type= int
        )
        return tf.constant(tokens, dtype= tf.int32)

    def _tf_sp_tokenize(self, line):
        tokens = tf.py_function(
            func= self._sp_tokenize,
            inp= [line],
            Tout= tf.int32
        )
        tokens.set_shape([None])
        return tokens
    
    def create(self, text_file: str, cache: bool, training: bool) -> tf.data.Dataset:
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
            shift= self.shift,
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
        if cache:
            ds = ds.cache()
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
        max_seq_len: int,
        embed_size: int,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.embed_size = embed_size

        if embed_size % 2 != 0:
            raise ValueError('embed_size must be even')

        self.supports_masking = True

    def build(self, input_shape):
        position = tf.range(self.max_seq_len, dtype= tf.float32)[:, tf.newaxis]
        div_term = tf.exp(
            tf.range(0, self.embed_size, 2, dtype= tf.float32)
            * (-tf.math.log(10000.0) / self.embed_size)
        )

        pe = tf.zeros((self.max_seq_len, self.embed_size), dtype= tf.float32)
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices= tf.reshape(tf.range(0, self.embed_size, 2), (-1, 1)),
            updates= tf.sin(position * div_term)
        )
        pe = tf.tensor_scatter_nd_update(
            pe,
            indices= tf.reshape(tf.range(1, self.embed_size, 2), (-1, 1)),
            updates= tf.cos(position * div_term)
        )

        pe = pe[tf.newaxis, ...]  # (1, max_seq_len, embed_size)

        self.pos_encodings = tf.cast(pe, self.compute_dtype)
        self.built = True

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encodings[:, :seq_len, :]

    def get_config(self):
        config = super().get_config()
        config.update({
            'max_seq_len': self.max_seq_len,
            'embed_size': self.embed_size,
        })
        return config


# ----------------------------
# Layer Normalization
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, *, epsilon: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape: tuple[int, ...]) -> None:
        dim = input_shape[-1],
        self.gamma = self.add_weight(
            name= 'gamma', 
            shape= dim,
            initializer= 'ones',
            trainable= True
        )
        
        self.beta = self.add_weight(
            name= 'beta', 
            shape= dim,
            initializer= 'zeros',
            trainable= True
        )
        super().build(input_shape)

    def call(self, X):
        X = tf.cast(X, self.compute_dtype)
        mean, variance = tf.nn.moments(X, axes= -1, keepdims= True)
        normalized = (X - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'epsilon': self.epsilon})
        return config


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

    def call(self, x):
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

    def call(self, x):
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

    def call(self, x, training: bool = False):
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
def perplexity(y_true, y_pred):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits= True
    )

    mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
    loss = loss * mask

    mean_loss = tf.reduce_sum(loss) / tf.reduce_sum(mask)
    return tf.exp(mean_loss)
