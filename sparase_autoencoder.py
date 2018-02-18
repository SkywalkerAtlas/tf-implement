import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data


def input_fn_gen(function, features_generator, labels=None, batch_size=None):
    classifier_x = []
    for result in features_generator:
        classifier_x.append(result['encoder_result'])
    # print(classifier_x)
    classifier_x = tf.convert_to_tensor(np.asarray(classifier_x))
    assert function == 'train' or function == 'predict' or function == 'eval', "function must be 'train', 'predict' or 'eval'"
    if function == 'train':
        data_set = tf.data.Dataset.from_tensor_slices((classifier_x, labels))
        data_set = data_set.shuffle(512).repeat().batch(batch_size)
    elif function == 'eval':
        data_set = tf.data.Dataset.from_tensor_slices((classifier_x, labels))
        data_set = data_set.batch(batch_size)
    else:
        data_set = tf.data.Dataset.from_tensor_slices(classifier_x).batch(batch_size)

    return data_set



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
    # features = dict(features)
    assert function == 'train' or function == 'predict' or function == 'eval', "function must be 'train', 'predict' or 'eval'"
    if function == 'train':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        data_set = data_set.shuffle(512).repeat().batch(batch_size)
    elif function == 'eval':
        data_set = tf.data.Dataset.from_tensor_slices((features, labels))
        data_set = data_set.batch(batch_size)
    else:
        data_set = tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

    return data_set


def sparase_autoencoder(features, labels, mode, params):
    """some note"""
    net = tf.layers.dense(features, features.shape[1], activation=None, name='INPUT')

    # define encoder
    encoder_net = net
    for index, units in enumerate(params['encoder_units']):
        encoder_net = tf.layers.dense(encoder_net, units, activation=tf.nn.sigmoid, name='encoder_'+str(index))

    # Define encoder output
    en_result = tf.layers.dense(encoder_net, params['encoder_result_units'], activation=tf.nn.sigmoid, name='ENCODER_OUTPUT')

    # define decoder
    decoder_net = en_result
    for index, units in enumerate(params['decoder_units']):
        decoder_net = tf.layers.dense(decoder_net, units, activation=tf.nn.sigmoid, name='decoder_'+str(index))

    # Define decoder result
    decoder_result = tf.layers.dense(decoder_net, features.shape[1], activation=tf.nn.sigmoid, name='DECODER_OUTPUT')

    # compute predictions
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'encoder_result': en_result
        }
        return tf.estimator.EstimatorSpec(mode, predictions)

    # compute loss
    loss = tf.losses.mean_squared_error(labels=labels, predictions=decoder_result)

    # compute evaluation metrics
    # accuracy = tf.metrics.accuracy(labels=labels, predictions=decoder_result, name='accuracy_op')
    # metrics = {'accuracy': accuracy}
    # tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def pic_classifier(features, labels, mode, params):
    """some note"""
    net = tf.layers.dense(features, features.shape[1], activation=None, name='INPUT')

    # define classifier network
    for index, units in enumerate(params['units']):
        net = tf.layers.dense(net, units, activation=tf.nn.sigmoid, name='hidden_'+str(index))

    logits = tf.layers.dense(net, params['n_classes'], activation=None, name='logits')

    predicted_class = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_class[:, tf.newaxis],
            'logits': logits
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute losses
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))

    # Compute evaluation metrics
    labels = tf.argmax(labels, 1)
    accuracy = tf.metrics.accuracy(labels, predictions=predicted_class, name='classifier_acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # create train op
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main(argv):
    mnist = input_data.read_data_sets('../MNIST_data/', one_hot=True)
    # mnist.train.next_batch()

    # Build the auto encoder
    auto_encoder = tf.estimator.Estimator(
        model_fn=sparase_autoencoder,
        model_dir='../log/SAE',
        params={
            'encoder_units': [],
            'encoder_result_units': 200,
            'decoder_units': [],
        }
    )

    # Train the model
    auto_encoder.train(
        input_fn=lambda : input_fn('train', mnist.train.images, mnist.train.images, batch_size=128),
        steps=3000
    )

    # # Evaluate the model
    # eval_result = auto_encoder.evaluate(
    #     input_fn=lambda : input_fn(
    #         function='eval',
    #         features=mnist.test.images[:20],
    #         labels=mnist.test.images[:20],
    #         batch_size=20
    #     )
    # )

    # Build mnist classifier
    mnist_classifier = tf.estimator.Estimator(
        model_fn=pic_classifier,
        model_dir='../log/classifier',
        params={
            'units': [128],
            'n_classes': 10,
        }
    )

    # Get training input data from previous auto encoder
    classifier_gen = auto_encoder.predict(
        input_fn=lambda : input_fn('predict', mnist.train.images, batch_size=mnist.train.images.shape[0])
    )


    # Train classifier
    mnist_classifier.train(
        input_fn=lambda : input_fn_gen('train', classifier_gen, mnist.train.labels, batch_size=128),
        steps=3000
    )

    # Get test data
    classifier_gen_test = auto_encoder.predict(
        input_fn=lambda :input_fn('predict', mnist.test.images, batch_size=mnist.train.images.shape[0])
    )

    # Evaluate classifier
    classifier_eval_result = mnist_classifier.evaluate(
        input_fn=lambda : input_fn_gen('eval', classifier_gen_test, mnist.test.labels, mnist.test.labels.shape[0])
    )

    print('\nTest accuracy:{accuracy: 0.3f}\n'.format(**classifier_eval_result))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main)
