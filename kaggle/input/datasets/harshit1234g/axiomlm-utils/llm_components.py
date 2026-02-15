import tensorflow as tf
import numpy as np
from sentencepiece import SentencePieceProcessor

# ----------------------------
# Tokenizer
# ----------------------------
def load_sp_tokenizer(path: str) -> SentencePieceProcessor:
    """
    Loads sentence piece tokenizer from the given path.

    Args:
        path (str): Path of the `*.model` file.

    Returns:
        SentencePieceProcessor: The loaded sentence piece tokenizer.
    """
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
        """
        An efficient dataset loading pipeline. `LMDatasetLoader` creates a uniform loading pipeline, to read text data, apply tokenization, and create batched instances of shape `(self.batch_size, self.seq_len)`.

        Args:
            tokenizer (SentencePieceProcessor): The sentence piece tokenizer to use for tokenizing text. Can be loaded using `load_sp_tokenizer(path)` function.
            shift (int): The amount of shift to use between instances.
            seq_len (int): Context size for the model, a single instance will have `seq_len` many tokens.
            batch_size (int): Batch size.
            shuffle_buffer (int): Shuffle buffer, only applied if `self.create(text_file= ..., training= True)`
        """
        self.tokenizer = tokenizer
        self.shift = shift
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.shuffle_buffer = shuffle_buffer

    def _sp_tokenize(self, line):
        """
        Creates tokens of the given line. Private function, not meant to be called directly.
        """
        tokens = self.tokenizer.encode(
            line.numpy().decode('utf-8'), 
            out_type= int
        )
        return tf.constant(tokens, dtype= tf.int32)

    def _tf_sp_tokenize(self, line):
        """
        `tf.py_function` wrapper for `self._sp_tokenize`. Private function, not meant to be called directly.
        """
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
        """
        Creates the `tf.data.Dataset` object.

        Args:
            text_file (str): Path of the text file.
            training (bool): If training is True, then it will apply `shuffle` and `repeat`.

        Returns:
            tf.data.Dataset: The created dataset object.
        """
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

        # [[t1], [t2], ...] -> [t1, t2, ...]
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
        """
        Can't use Batch Normalization because data is sequential, so we use Layer Normalization. Layer Normalization (LayerNorm) is a technique used in deep learning to stabilize and accelerate the training of neural networks by normalizing the inputs across the feature dimension for each data sample independently.

        Args:
            epsilon (float, optional): The value to add in variance, so that zero division can be avoided. Defaults to 1e-5.
        """
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
        """
        Multi-Headed Self-Attention layer for a decoder-only Transformer. This module implements scaled dot-product multi-head self-attention with optional KV caching for efficient autoregressive inference.

        Args:
            n_embeds (int): Embedding dimension of the model (also referred to as `d_model`). Input and output tensors have shape `(B, T, n_embeds)`.
            n_heads (int): Number of attention heads. The `n_embeds` must be divisible by `n_heads`.

        Raises:
            ValueError: If `n_embeds` is not divisible by `n_heads`.

        Notes:
        - Each head operates on `head_dim = n_embeds // n_heads`.
        - Queries, Keys, and Values are computed using a single projection
          layer and then split.
        - During training, a full causal mask (lower triangular) is applied.
        - During autoregressive generation, KV cache can be supplied to
          avoid recomputing previous keys and values.
        """
        super().__init__(**kwargs)

        if n_embeds % n_heads != 0:
            raise ValueError('n_embeds must be divisible by n_heads')

        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.head_dim = n_embeds // n_heads
        self.scale = self.head_dim ** -0.5     # scaling factor for stability

        # Single linear projection for Q, K, V (more efficient than 3 separate layers)
        # Output shape: (B, T, 3 * n_embeds)
        self.qkv = tf.keras.layers.Dense(3 * n_embeds, use_bias= False)

        # Final projection after concatenating all heads
        # Output shape: (B, T, n_embeds)
        self.proj = tf.keras.layers.Dense(n_embeds)

    def call(
        self, 
        x,
        *,
        past_k=None,
        past_v=None,
        use_cache: bool = False,
        training: bool = False
    ):
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        qkv = self.qkv(x)                       # (B, T, 3 * C)
        q, k, v = tf.split(qkv, 3, axis= -1)    # Each have shape of (B, T, C) 

        # reshaping to multihead format, (B, T, H, D)
        q = tf.reshape(q, (B, T, self.n_heads, self.head_dim))
        k = tf.reshape(k, (B, T, self.n_heads, self.head_dim))
        v = tf.reshape(v, (B, T, self.n_heads, self.head_dim))

        q = tf.transpose(q, (0, 2, 1, 3))  # (B, H, T, D)
        k = tf.transpose(k, (0, 2, 1, 3))
        v = tf.transpose(v, (0, 2, 1, 3))

        # KV cache
        if past_k is not None and past_v is not None:
            k = tf.concat([past_k, k], axis= 2)  # concat on sequence dim
            v = tf.concat([past_v, v], axis= 2)

        present_k = k
        present_v = v

        # (Q @ K^T) / d_k
        att = tf.matmul(q, k, transpose_b= True) * self.scale

        # Training mode, full causal masking
        if past_k is None:
            mask = tf.linalg.band_part(tf.ones((T, T)), -1, 0)
            mask = tf.reshape(mask, (1, 1, T, T))
            att = tf.where(mask == 0, -1e9, att)

        # applying softmax (attention over keys)
        att = tf.nn.softmax(att, axis= -1)

        out = tf.matmul(att, v)  # (B, H, T, D)
        out = tf.transpose(out, (0, 2, 1, 3))
        out = tf.reshape(out, (B, T, self.n_embeds))   # (B, T, C)
        out = self.proj(out)

        if use_cache:
            return out, present_k, present_v

        return out
    
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
        """
        MLP part of the Transformer block.

        Args:
            n_embed (int): Embedding dimension (d_model).
            activation (str, optional): Activatin function to use for first dense layer. Defaults to 'gelu'.
        """
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
            kernel_initializer= tf.keras.initializers.GlorotUniform(),
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
        """
        Single Transformer block (pre-norm) used in decoder-only GPT model.

        This block consists of:
        1. LayerNorm
        2. Multi-Head Self-Attention
        3. Residual connection
        4. LayerNorm
        5. FeedForward network
        6. Residual connection

        Args:
            n_embeds (int): Embedding dimensions. Input and output tensors have shape `(B, T, n_embeds)`.
            n_heads (int): Number of attention heads in the self-attention layer.

        Notes:
        - This implementation uses Pre-Norm architecture:
            - x = x + Attention(LN(x))
            - x = x + FFN(LN(x))
        - Supports optional KV caching for autoregressive decoding.
        """
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

    def call(
        self, 
        x, 
        *,
        past_k=None,
        past_v=None,
        use_cache: bool = False,
        training: bool = False
    ):
        # using cache
        if use_cache:
            attn_out, new_k, new_v = self.attn(
                self.ln1(x),
                past_k= past_k,
                past_v= past_v,
                use_cache= True,
                training= training
            )
            x = x + attn_out
            x = x + self.ffwd(self.ln2(x), training=training)
            return x, new_k, new_v
        
        # Standard full-sequence training path
        else:
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
        """
        Perplexity is a key evaluation metric for language models that measures how well a probability model predicts a sample. It is defined as the exponentiated average negative log-likelihood of a sequence.

        Args:
            name (str, optional): Name of the metric. Defaults to 'perplexity'.
        """
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
        """
        Decoder-only GPT language model.

        Architecture:
        - Token embedding
        - Learned positional embedding
        - Stack of Transformer blocks
        - Final LayerNorm
        - Weight-tied output projection

        Args:
            vocab_size (int): Vocabulary size for token embedding.
            seq_len (int): Maximum context length supported by the model.
            n_embeds (int): Embedding dimension (d_model).
            n_heads (int): Number of attention heads per block.
            n_blocks (int): Number of stacked Transformer blocks.

        Notes:
        - Uses learned positional embeddings.
        - Uses weight tying between input embedding and output projection.
        - Supports KV caching for efficient autoregressive generation.
        """
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embeds = n_embeds
        self.n_heads = n_heads
        self.n_blocks = n_blocks

        # Token embedding: (B, T) -> (B, T, C)
        self.token_emb = tf.keras.layers.Embedding(self.vocab_size, self.n_embeds)

        # Learned positional embeddings
        # Positions range: [0, seq_len)
        self.pos_emb = tf.keras.layers.Embedding(self.seq_len, self.n_embeds)
        self.position_ids = tf.range(self.seq_len)

        # Stack of Transformer blocks
        self.blocks = [
            TransformerBlock(
                n_embeds= self.n_embeds,
                n_heads= self.n_heads
            )
            for _ in range(self.n_blocks)
        ]

        # Final LayerNorm before output projection
        self.ln_f = LayerNormalization()

    def call(
        self, 
        input_ids, 
        past=None,
        use_cache: bool = False,
        training: bool = False
    ):
        # Current sequence length (may be 1 during generation)
        seq_len = tf.shape(input_ids)[1]

        token_embeddings = self.token_emb(input_ids)

        # Training or first forward pass
        if past is None:
            positions = self.position_ids[:seq_len]

        else:
            # During generation, shift positions by past length
            # past[0][0] -> key tensor of layer 0
            # shape: (B, H, T_past, D)
            past_len = tf.shape(past[0][0])[2]
            positions = self.position_ids[past_len: past_len + seq_len]

        pos_embeddings = self.pos_emb(positions)    # (T, C)
        # Broadcast addition -> (B, T, C)
        x = token_embeddings + pos_embeddings

        new_past = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            if use_cache:
                # Extract per-layer cache
                past_k, past_v = (None, None) if past is None else past[i]
                x, k, v = block(
                    x,
                    past_k= past_k,
                    past_v= past_v,
                    use_cache= True,
                    training= training
                )
                new_past.append((k, v))

            else:
                x = block(x, training= training)

        x = self.ln_f(x)

        # weight tying, Unembedding matrix is transpose of embedding
        logits = tf.matmul(
            x,
            self.token_emb.embeddings,
            transpose_b= True
        )

        if use_cache:
            return logits, new_past

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
        """
        Linear warmup followed by cosine decay learning rate schedule.

        Args:
            base_lr (float): Peak learning rate reached after warmup.
            warmup_steps (int): Number of steps for linear warmup from 0 to `base_lr`.
            total_steps (int): Total number of training steps.
            min_lr_ratio (float, optional): Final LR = `base_lr * min_lr_ratio`. Defaults to 0.1.
        """
        super().__init__()
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = base_lr * min_lr_ratio

    def __call__(self, step):
        step = tf.cast(step, tf.float32)

        lr = tf.cond(
            step < self.warmup_steps,
            #  Linear warmup: lr increases from 0 -> base_lr
            lambda: self.base_lr * step / self.warmup_steps,
            # Cosine decay: base_lr -> min_lr
            lambda: self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + tf.cos(
                    tf.constant(np.pi) *
                    (step - self.warmup_steps) /
                    (self.total_steps - self.warmup_steps)
                )
            )
        )
        return lr
