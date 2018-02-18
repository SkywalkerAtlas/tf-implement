import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def input_fn(function, features, labels=None, batch_size=None):
    """Custom input function to input data for training, evaluation or prediction step


    Args:
        function: selected function, must be train, predict or eval
        features: input_x
        batch_size: batch size to return
        labels: input_y

    Returns:
        a Dataset
    """
    features = dict(features)
    assert function == 'train' or 'predict' or 'eval', "function must be 'train', 'predict' or 'eval'"
    if function == 'train':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        data_set = data_set.shuffle(1024).repeat().batch(batch_size)
    elif function == 'eval':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
    else:
        data_set = tf.data.Dataset.from_tensor_slices(features)

    return data_set


def sparase_autoencoder(features, labels, mode, params):
    """some note"""
    net = tf.feature_column.input_layer(features ,params['feature_columns'])

    # define encoder
    for units in params['encoder_units']:
        encoder_net = tf.layers.dense(net, units, activation=tf.nn.sigmoid)

    # define decoder
    for units in params['decoder_units']:
        decoder_net = tf.layers.dense(encoder_net, units, activation=tf.nn.sigmoid)

    # compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'encoder_result': encoder_net
        }
        return tf.estimator.EstimatorSpec(mode, predictions)

    # compute loss
    loss = tf.losses.mean_squared_error(labels, decoder_net)

    # compute evaluation metrics
    accuracy = tf.metrics.accuracy(labels, predictions=decoder_net, name='accuracy_op')
    metrics = {'accuracy', accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss, train_op=train_op)


def pic_classifier(features, labels, mode, params):
    """some note"""
    net = tf.feature_column.input_layer(features, params['feature_column'])

    # define classifier network
    for units in params['units']:
        net = tf.layers.dense(net, units, activation=tf.nn.sigmoid)

    logits = tf.layers.dense(net, params['n_class'], activation=None)

    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class[:, tf.newaxis],
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels, logits)

    accuracy = tf.metrics.accuracy(labels, predictions=predicted_class, name='classifier_acc_op')
    metrics = {'accuracy', accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss, eval_metric_ops=metrics)

    # create train op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss, train_op=train_op)


def main():
    pass

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
