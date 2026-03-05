import re
from typing import Generator
import tensorflow as tf
import numpy as np
from sentencepiece import SentencePieceProcessor

# -------------------------------
# Tokenize file and save as .npy
# -------------------------------
def write_tokens_npy(
    sp_model_path: str,
    text_file_path: str,
    output_path: str,
    *,
    dtype: np.dtype = np.uint16
) -> None:
    """
    Reads a text file and then tokenize it line-by-line and then saves it as a `*.npy` file.

    Args:
        sp_model_path (str): Path to the sentencepiece tokenizer.
        text_file_path (str): Path to the text file.
        output_path (str): Path to the output file.
        dtype (np.dtype, optional): Datatype of the resulting `*.npy` file. Defaults to np.uint16.
    """
    # loading tokenizer
    sp = SentencePieceProcessor()
    sp.load(sp_model_path)

    tokens = []

    # reading the file line-by-line and then encoding and extending
    with open(text_file_path, 'r', encoding= 'utf-8') as f:
        for line in f:
            ids = sp.encode(line.strip(), out_type= int)
            tokens.extend(ids)

    tokens = np.array(tokens, dtype= dtype)
    np.save(output_path, tokens)   # saves the file as continous token stream

    print(f'Saved {len(tokens)} tokens to {output_path}')

# ----------------------------
# Loading data
# ----------------------------
def create_dataset_from_npy(
    npy_path: str,
    seq_len: int,
    batch_size: int,
    shift: int,
    shuffle_buffer: int,
    training: bool
) -> tf.data.Dataset:
    """
    Creates a loading pipeline, to read text data, apply tokenization, and create batched instances of shape `(batch_size, seq_len)`.

    Args:
        npy_path (str): Path to the `*.npy` file.
        seq_len (int): Context size of the model.
        batch_size (int): Batch size.
        shift (int): The amount of shift for creating windows, use 1 if want dense dataset, else use `seq_len`.
        shuffle_buffer (int): Shuffle buffer, only applied when training is True.
        training (bool): True if training dataset, else False.

    Returns:
        tf.data.Dataset: The created dataset object.
    """
    # loading mmap
    tokens = np.load(npy_path)
    tokens = tf.convert_to_tensor(tokens, dtype= tf.int32)

    ds = tf.data.Dataset.from_tensor_slices(tokens)

    # create sliding windows of tokens
    ds = ds.window(
        seq_len + 1, 
        shift= shift, 
        drop_remainder= True
    )
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))

    # Split into (input, target)
    ds = ds.map(
        lambda x: (x[:-1], x[1:]), 
        num_parallel_calls= tf.data.AUTOTUNE
    )

    if training:
        ds = ds.shuffle(shuffle_buffer)
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder= True)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def sft_pair_generator(features_path: str, labels_path: str) -> Generator:
    """
    Generator that yields (input_ids, labels)
    """
    features = np.load(features_path, allow_pickle= True)
    labels = np.load(labels_path, allow_pickle= True)

    assert len(features) == len(labels), 'Features and labels size mismatch'

    for x, y in zip(features, labels):
        yield (
            np.array(x, dtype= np.int32),
            np.array(y, dtype= np.int32),
        )

def load_sft_dataset(
    features_path: str,
    labels_path: str,
    pad_token_id: int,
    batch_size: int,
    shuffle_buffer: int
) -> tf.data.Dataset:
    """
    Loads the preprocessed sft dataset.

    Args:
        features_path (str): Path to the `processed_features.npy` file.
        labels_path (str): Path to the `processed_labels.npy` file.
        pad_token_id (int): Pad token id of the tokenizer.
        batch_size (int): Batch size.
        shuffle_buffer (int): Shuffle buffer.

    Returns:
        tf.data.Dataset: The created dataset object.
    """
    output_signature = (
        tf.TensorSpec(shape= (None,), dtype= tf.int32),
        tf.TensorSpec(shape= (None,), dtype= tf.int32),
    )

    ds = tf.data.Dataset.from_generator(
        lambda: sft_pair_generator(features_path, labels_path),
        output_signature= output_signature,
    )

    ds = ds.shuffle(shuffle_buffer)

    ds = ds.padded_batch(
        batch_size,
        padded_shapes= ([None], [None]),
        padding_values= (pad_token_id, -100)
    )

    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

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
@tf.keras.utils.register_keras_serializable()
class WarmupCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, 
        base_lr: float, 
        warmup_steps: int, 
        total_steps: int, 
        min_lr_ratio: float = 0.1,
        **kwargs
    ) -> None:
        """
        Linear warmup followed by cosine decay learning rate schedule.

        Args:
            base_lr (float): Peak learning rate reached after warmup.
            warmup_steps (int): Number of steps for linear warmup from 0 to `base_lr`.
            total_steps (int): Total number of training steps.
            min_lr_ratio (float, optional): Final LR = `base_lr * min_lr_ratio`. Defaults to 0.1.
        """
        super().__init__(**kwargs)
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
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
    
    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'min_lr_ratio': self.min_lr_ratio
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# ----------------------------
# Metric
# ----------------------------
@tf.keras.utils.register_keras_serializable()
class Perplexity(tf.keras.metrics.Metric):
    def __init__(
        self, 
        pad_id: int, 
        ignore_index: int | None = None, 
        name: str = 'ppl', **kwargs
    ):
        """
        Perplexity is a key evaluation metric for language models that measures how well a probability model predicts a sample. It is defined as the exponentiated average negative log-likelihood of a sequence.

        Args:
            pad_id (int): <pad> token id of the tokenizer.
            name (str, optional): Name of the metric. Defaults to 'perplexity'.
            ignore_index (int, optional): The class to ignore for calculating loss. Defaults to None.
        """
        super().__init__(name= name, **kwargs)

        self.pad_id = pad_id
        self.ignore_index = ignore_index

        self.total_loss = self.add_weight(
            name= 'total_loss', 
            initializer= 'zeros',
            dtype= tf.float32
        )
        self.total_tokens = self.add_weight(
            name= 'total_tokens', 
            initializer= 'zeros',
            dtype= tf.float32
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true, 
            y_pred, 
            from_logits= True,
            ignore_class= self.ignore_index
        )

        mask = tf.cast(tf.not_equal(y_true, self.pad_id), tf.float32)
        loss = loss * mask

        self.total_loss.assign_add(tf.reduce_sum(loss))
        self.total_tokens.assign_add(tf.reduce_sum(mask))

    def result(self):
        return tf.exp(self.total_loss / tf.maximum(self.total_tokens, 1.0))

    def reset_state(self):
        self.total_loss.assign(0.0)
        self.total_tokens.assign(0.0)

    def get_config(self):
        config = super().get_config()
        config.update({
            'pad_id': self.pad_id,
            'ignore_index': self.ignore_index
        })
        return config

# ----------------------------------
# Text generation & Chat completion
# ----------------------------------
def clean_text(text: str) -> str:
    # removing wikitext artifacts
    text = text.replace('@,@', ',')
    text = text.replace('@-@', '-')
    text = text.replace('@.@', '.')

    # remvoing <unk>, just in case
    text = text.replace('<unk>', '')

    # remove section headers
    text = re.sub(r'=+\s*[^=]+\s*=+', '', text)

    # fixing possessives
    text = re.sub(r"\s+'s", "'s", text)

    # fixing parentheses spacing
    text = re.sub(r'\(\s+', '(', text)
    text = re.sub(r'\s+\)', ')', text)

    # fixing space in quotation
    text = re.sub(r'\s+"', '"', text)
    text = re.sub(r'"\s+', '"', text)
    text = re.sub(r"\s+'", "'", text)
    text = re.sub(r"'\s+", "'", text)

    # removing extra spaces before punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)

    # Collapse multiple spaces
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


def top_k_logits(logits, k):
    values, _ = tf.math.top_k(logits, k= k)
    min_values = values[:, -1, tf.newaxis]
    return tf.where(logits < min_values, -1e9, logits)


def apply_frequency_penalty(logits, generated_ids, penalty: float):
    ids = tf.constant(generated_ids, dtype= tf.int32)
    counts = tf.math.bincount(ids, minlength= logits.shape[-1])
    logits -= penalty * tf.cast(counts, logits.dtype)
    return logits


def generate_text(
    model: tf.keras.Model,
    tokenizer: SentencePieceProcessor,
    prompt: str,
    *,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: int = 50,
    freq_penalty: float = 0.05
) -> str:
    if max_new_tokens > model.seq_len:
        raise ValueError(f'Currently the model has fixed length context window, so cannot generate text with more than {model.seq_len} tokens.')
    
    # encoding prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = tf.constant([input_ids], dtype= tf.int32)

    # truncate if prompt exceeds context window
    if input_ids.shape[1] > model.seq_len:
        input_ids = input_ids[:, -model.seq_len:]

    # full prompt, first forward pass
    logits, past = model(
        input_ids,
        past= None,
        use_cache= True,
        training= False
    )
    generated_ids = input_ids.numpy().tolist()[0]

    # Only feed last token from now on
    next_token = input_ids[:, -1:]

    for _ in range(max_new_tokens):
        if len(generated_ids) >= model.seq_len:
            break

        logits, past = model(
            next_token,
            past= past,
            use_cache= True,
            training= False
        )

        logits = logits[:, -1, :]

        # Prevent <unk> sampling
        unk_id = tokenizer.unk_id()
        logits = tf.tensor_scatter_nd_update(
            logits,
            indices= [[0, unk_id]],
            updates= [-1e9]
        )

        logits = apply_frequency_penalty(logits, generated_ids, freq_penalty)

        if temperature != 1.0:
            logits = logits / temperature

        logits = top_k_logits(logits, k= top_k)
        next_token = tf.random.categorical(logits, num_samples= 1)

        token_id = int(next_token.numpy()[0][0])
        generated_ids.append(token_id)

        eos_id = tokenizer.eos_id()
        if eos_id is not None and token_id == eos_id:
            break

    text = tokenizer.decode(generated_ids)
    return clean_text(text)
