import tensorflow as tf
from sentencepiece import SentencePieceProcessor


# ----------------------------
# Constants
# ----------------------------
SEQUENCE_LEN = 256
BATCH_SIZE = 8
SHUFFLE_BUFFER = 10_000


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