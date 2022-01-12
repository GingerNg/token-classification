from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood
import tensorflow as tf
from tensorflow.keras import backend as K

def get_negative_log_likelihood(y_true, y_pred):

    # _, potentials, sequence_length, chain_kernel = y_pred
    # print(y_pred.numpy())
    print(type(y_pred))
    _, potentials, sequence_length, chain_kernel = y_pred
    # TODO: remove typing cast
    potentials = tf.keras.backend.cast(potentials, tf.float32)
    y_true = tf.keras.backend.cast(y_true, tf.int32)
    sequence_length = tf.keras.backend.cast(sequence_length,tf.int32)
    # self.chain_kernel = tf.keras.backend.cast(self.chain_kernel,
    #                                           tf.float32)

    log_likelihood, _ = crf_log_likelihood(potentials, y_true, sequence_length, chain_kernel)

    return tf.reduce_mean(-log_likelihood)

def crf_loss(y_true, y_pred):
    # we don't use y_pred, but caller pass it anyway, ignore it
    return get_negative_log_likelihood(y_true, y_pred)

def unpack_data(data):
    if len(data) == 2:
        return data[0], data[1], None
    elif len(data) == 3:
        return data
    else:
        raise TypeError("Expected data to be a tuple of size 2 or 3.")

class ModelWithCRFLoss(tf.keras.Model):
    """Wrapper around the base model for custom training logic."""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def call(self, inputs):
        return self.base_model(inputs)

    def compute_loss(self, y_true, y_pred, sample_weight, training=False):

        # print(type(y_pred))
        _, potentials, sequence_length, chain_kernel = y_pred
        # potentials = potentials.numpy()
        # chain_kernel = chain_kernel.numpy()
        # sequence_length = sequence_length.numpy()
        # print(sequence_length)
        # print(potentials.shape, sequence_length.shape, chain_kernel.shape)
        # print(y.shape)
        # we now add the CRF loss:
        crf_loss,_ = crf_log_likelihood(potentials, y_true, sequence_length, chain_kernel)
        crf_loss = -crf_loss
        if sample_weight is not None:
            crf_loss = crf_loss * sample_weight

        # return tf.reduce_mean(crf_loss), sum(self.losses)
        return tf.reduce_mean(crf_loss)

    def compute_accuracy(self, y_true, y_pred):
        y_pred, _, _, _, = y_pred
        judge = tf.keras.backend.cast(
            tf.keras.backend.equal(y_pred, y_true), tf.keras.backend.floatx())
        if self.mask is None:
            return tf.keras.backend.mean(judge)
        else:
            mask = tf.keras.backend.cast(self.mask, tf.keras.backend.floatx())
            return (tf.keras.backend.sum(judge * mask) /
                    tf.keras.backend.sum(mask))

    def train_step(self, data):
        x, y_true, sample_weight = unpack_data(data)
        y_pred = self(x, training=True)

        with tf.GradientTape() as tape:
            crf_loss = self.compute_loss(
                y_true, y_pred, sample_weight, training=True
            )
            total_loss = crf_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # acc = self.compute_accuracy(y_true, y_pred)

        return {"crf_loss": crf_loss, "crf_acc": 0}

    def test_step(self, data):
        x, y_true, sample_weight = unpack_data(data)
        y_pred = self(x, training=False)
        crf_loss = self.compute_loss(
            y_true, y_pred, sample_weight, training=False
        )
        return {"crf_loss_val": crf_loss, "crf_acc": 0}

from tensorflow_addons.layers import CRF
# @tf.keras.utils.register_keras_serializable(package="Addons")
class MyCRF(CRF):
    def call(self, inputs, mask=None):
        # mask: Tensor(shape=(batch_size, sequence_length), dtype=bool) or None
        self.mask = mask
        if mask is not None:
            if tf.keras.backend.ndim(mask) != 2:
                raise ValueError("Input mask to CRF must have dim 2 if not None")

        if mask is not None:
            # left padding of mask is not supported, due the underline CRF function
            # detect it and report it to user
            left_boundary_mask = self._compute_mask_left_boundary(mask)
            first_mask = left_boundary_mask[:, 0]
            if first_mask is not None and tf.executing_eagerly():
                no_left_padding = tf.math.reduce_all(first_mask)
                left_padding = not no_left_padding
                if left_padding:
                    raise NotImplementedError(
                        "Currently, CRF layer do not support left padding"
                    )

        self.potentials = self._dense_layer(inputs)

        # appending boundary probability info
        if self.use_boundary:
            self.potentials = self.add_boundary_energy(
                self.potentials, mask, self.left_boundary, self.right_boundary
            )

        self.sequence_length = self._get_sequence_length(inputs, mask)

        decoded_sequence, _ = self.get_viterbi_decoding(self.potentials, self.sequence_length)

        return decoded_sequence
        # return [self.decoded_sequence, self.potentials, self.sequence_length, self.chain_kernel]

    def get_negative_log_likelihood(self, y_true):
        # TODO: remove typing cast
        self.potentials = tf.keras.backend.cast(self.potentials, tf.float32)
        y_true = tf.keras.backend.cast(y_true, tf.int32)
        self.sequence_length = tf.keras.backend.cast(self.sequence_length,
                                                     tf.int32)
        # self.chain_kernel = tf.keras.backend.cast(self.chain_kernel,
        #                                           tf.float32)

        log_likelihood, _ = crf_log_likelihood(
            self.potentials, y_true, self.sequence_length, self.chain_kernel)

        return  tf.reduce_mean(-log_likelihood)

    def crf_loss(self, y_true, y_pred):
        # we don't use y_pred, but caller pass it anyway, ignore it
        return self.get_negative_log_likelihood(y_true)

    def crf_accuracy(self, y_true, y_pred):
        # print(y_pred.shape)
        judge = tf.keras.backend.cast(
            tf.keras.backend.equal(y_pred, y_true), tf.keras.backend.floatx())
        # print(judge)
        if self.mask is None:
            return tf.keras.backend.mean(judge)
        else:
            mask = tf.keras.backend.cast(self.mask, tf.keras.backend.floatx())
            return (tf.keras.backend.sum(judge * mask) /
                    tf.keras.backend.sum(mask))
