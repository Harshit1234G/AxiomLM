import tensorflow as tf
import numpy as np
from sentencepiece import SentencePieceProcessor

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
        seq_len: int,
        batch_size: int,
        shuffle_buffer: int
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
    
    def create(
        self, 
        text_file: str,
        training: bool
    ) -> tf.data.Dataset:
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
        ds = ds.prefetch(AUTOTUNE)
        
        return ds
 

# ----------------------------
# Layer Normalization
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self, *, epsilon: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        dim = input_shape[-1:]
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

    def get_config(self):
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
        if n_embeds % n_heads != 0:
            raise ValueError('n_embeds must be divisible by n_heads')

        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.head_dim = n_embeds // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = tf.keras.layers.Dense(3 * n_embeds, use_bias= False)
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
        out = tf.reshape(out, (B, T, self.n_embeds))

        return self.proj(out)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'n_embeds': self.n_embeds,
            'n_heads': self.n_heads,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class FeedForward(tf.keras.layers.Layer):
    def __init__(
        self, 
        n_embed: int,
        *,
        activation: str = 'gelu',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.n_embed = n_embed
        self.activation = tf.keras.activations.get(activation)

        self.fc1 = tf.keras.layers.Dense(
            4 * n_embed,
            activation= self.activation,
            kernel_initializer= tf.keras.initializers.GlorotUniform(),
        )
        self.fc2 = tf.keras.layers.Dense(
            n_embed,
            kernel_initializer=tf.keras.initializers.GlorotUniform(),
        )

    def call(self, x):
        return self.fc2(self.fc1(x))
    
    def get_config(self):
        config = super().get_config()
        config.update({'n_embed': self.n_embed})
        return config
    

@tf.keras.utils.register_keras_serializable()
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(
        self,
        n_embeds: int,
        n_heads: int,
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

    def get_config(self):
        config = super().get_config()
        config.update({
            'n_embeds': self.n_embeds,
            'n_heads': self.n_heads,
        })
        return config

# ----------------------------
# Metric
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Perplexity(tf.keras.metrics.Metric):
    def __init__(self, name: str = 'perplexity', **kwargs):
        super().__init__(name= name, **kwargs)

        self.total_loss = self.add_weight(
            name= 'total_loss', 
            initializer= 'zeros'
        )
        self.total_tokens = self.add_weight(
            name= 'total_tokens', 
            initializer= 'zeros'
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, 
            y_pred, 
            from_logits= True
        )

        mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
        loss = loss * mask

        self.total_loss.assign_add(tf.reduce_sum(loss))
        self.total_tokens.assign_add(tf.reduce_sum(mask))

    def result(self):
        return tf.exp(self.total_loss / self.total_tokens)

    def reset_states(self):
        self.total_loss.assign(0.0)
        self.total_tokens.assign(0.0)

# ----------------------------------
# Generative Pretrained Transformer
# ----------------------------------
@tf.keras.utils.register_keras_serializable()
class GPT(tf.keras.Model):
    def __init__(
        self,
        *,
        vocab_size: int,
        seq_len: int,
        n_embeds: int,
        n_heads: int,
        n_blocks: int,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        self.token_emb = tf.keras.layers.Embedding(self.vocab_size, self.n_embeds)
        self.pos_emb = tf.keras.layers.Embedding(self.seq_len, self.n_embeds)
        self.position_ids = tf.range(self.seq_len)

        self.blocks = [
            TransformerBlock(
                n_embeds= self.n_embeds,
                n_heads= self.n_heads
            )
            for _ in range(self.n_blocks)
        ]

        self.ln_f = LayerNormalization()

    def call(self, input_ids, training: bool = False):
        seq_len = tf.shape(input_ids)[1]

        token_embeddings = self.token_emb(input_ids)
        positions = self.position_ids[:seq_len]
        pos_embeddings = self.pos_emb(positions)
        x = token_embeddings + pos_embeddings

        for block in self.blocks:
            x = block(x, training= training)

        x = self.ln_f(x)

        # weight tying, Unembedding matrix is transpose of embedding
        logits = tf.matmul(
            x,
            self.token_emb.embeddings,
            transpose_b= True
        )
        return logits

    def get_config(self):
        config = super().get_config()
        config.update({
            'vocab_size': self.vocab_size,
            'seq_len': self.seq_len,
            'n_embeds': self.n_embeds,
            'n_heads': self.n_heads,
            'n_blocks': self.n_blocks,
        })
        return config


# ----------------------------
# LR Schedule
# ----------------------------
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, 
        base_lr: float, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr_ratio: float = 0.1
    ) -> None:
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        # Warmup
        lr = tf.cond(
            step < self.warmup_steps,
            lambda: self.base_lr * step / self.warmup_steps,
            lambda: self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + tf.cos(
                    tf.constant(np.pi) *
                    (step - self.warmup_steps) /
                    (self.total_steps - self.warmup_steps)
                )
            )
        )
        return lr
