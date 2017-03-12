# -*- encoding: utf-8 -*-

import numpy as np

import tensorflow as tf


#tensorboard --logdir=/Users/daniilmann/investparty/code/python/27/Futures/logs

def QL(data, layers, lr=.001, opt='adam', nb_batch=8, nb_epoch=1000):

    learn_X, learn_Y = data[0][0], data[0][1]
    valid_X, valid_Y = data[1][0], data[1][1]
    test_X, test_Y = data[2][0], data[2][1]

    print 'learn ', learn_X.shape, learn_Y.shape
    print 'valid ', valid_X.shape, valid_Y.shape
    print 'test ', test_X.shape, test_Y.shape

    lrs = layers
    layers = [learn_X.shape[-1]]
    layers.extend(lrs)
    print 'layers ', layers

    input_layer     = tf.placeholder(tf.float32, [None, layers[0]])
    output_layer    = tf.placeholder(tf.float32, [None, learn_Y.shape[-1]])

    #gamma  = tf.constant tf.Variable(tf.ones([1, learn_Y.shape[-1] - 1]), name='mu')
    l2 = tf.constant(.0000, tf.float32)
    ns = []

    out, o_h = [], []
    weights, w_h = [], []
    biases, b_h = [], []
    for i in range(1, len(layers)):
        ns.append(tf.constant(2. * layers[i]))
        with tf.name_scope('hidden_' + str(i)) as scope:
            weights.append(tf.Variable(tf.truncated_normal([layers[i-1], layers[i]], stddev=1.)))
            w_h.append(tf.histogram_summary('hweights_' + str(i), weights[-1]))
            biases.append(tf.Variable(tf.truncated_normal([layers[i]], stddev=.00001)))
            b_h.append(tf.histogram_summary('hbiases_' + str(i), biases[-1]))
            out.append(tf.add(tf.matmul(out[-1] if len(out) > 0 else input_layer, weights[-1]), biases[-1]))
            o_h.append(tf.histogram_summary('outs_' + str(i), out[-1]))

    with tf.name_scope('model') as scope:
        output = tf.squeeze(tf.tanh(out[-1])) #tf.sub(tf.to_float(tf.argmax(tf.nn.softmax(out[-1]), 1)), tf.constant(2., tf.float32))
        output_h = tf.histogram_summary('model', output)

    with tf.name_scope('cost') as scope:
        R1 = tf.reduce_sum(tf.mul(output, tf.squeeze(tf.slice(output_layer, [0, 0], [nb_batch, 1]))))
        r1_s = tf.scalar_summary('R1', R1)
        R2 = tf.reduce_sum(tf.abs(tf.slice(output_layer, [0, 1], [nb_batch, learn_Y.shape[-1] - 1])))
        r2_s = tf.scalar_summary('R2', R2)
        lr = tf.constant(.1, tf.float32)
        err = tf.mul(lr, tf.add(R1, R2))
        err = tf.add(err, tf.mul(lr, tf.sub(tf.add(R1, R2), err)))
        err_s = tf.scalar_summary('err_bl2', err)
        l2s = tf.constant(0.)
        for i in range(len(weights)):
            l2s -= l2 * tf.nn.l2_loss(weights[i]) - l2 * tf.nn.l2_loss(biases[i])
        l2_s = tf.scalar_summary('l2', l2s)
        costf = err + l2s
        costf_s = tf.scalar_summary('cost', costf)

    # var_list = weights
    # var_list.extend(biases)



    if opt == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-8)
    elif opt == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(lr)

    train_step = opt.minimize(-costf)

    result = tf.reduce_sum(tf.reduce_sum(output_layer, 1) * tf.sign(output))


    #rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(output_layer, output))))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('Futures/logs', graph_def=sess.graph_def)

    for i in range(nb_epoch):
        ix = np.random.randint(0, learn_X.shape[0], size=nb_batch)
        lx, ly = learn_X[ix], learn_Y[ix]
        ix = np.random.randint(0, valid_X.shape[0], size=nb_batch)
        vx, vy = valid_X[ix], valid_Y[ix]
        l2s.eval(feed_dict={input_layer: lx, output_layer: ly})
        e1 = err.eval(feed_dict={input_layer: lx, output_layer: ly})
        e2 = costf.eval(feed_dict={input_layer: lx, output_layer: ly})
        train_step.run(feed_dict={input_layer : lx, output_layer : ly})
        # summary_str = merged_summary_op.eval(feed_dict={input_layer : lx, output_layer : ly})
        # summary_writer.add_summary(summary_str, i)
        if i % 100 == 0:
            #print np.mean(weights[-1].eval()), np.std(weights[-1].eval())
            print 'step %s || %.3f || %.3f || %.3f' % (i, e1, e2, result.eval(feed_dict={input_layer : valid_X, output_layer : valid_Y}))
    print result.eval(feed_dict={input_layer : test_X, output_layer : test_Y})
    res = output.eval(feed_dict={input_layer : test_X, output_layer : test_Y})


    sess.close()

    return res


def DQN(data, layers, lr=.001, opt='adam', nb_batch=8, nb_epoch=1000):

    learn_X, learn_Y = data[0][0], data[0][1]
    valid_X, valid_Y = data[1][0], data[1][1]
    test_X, test_Y = data[2][0], data[2][1]

    print 'learn ', learn_X.shape, learn_Y.shape
    print 'valid ', valid_X.shape, valid_Y.shape
    print 'test ', test_X.shape, test_Y.shape

    lrs = layers
    layers = [learn_X.shape[-1]]
    layers.extend(lrs)
    print 'layers ', layers

    input_layer     = tf.placeholder(tf.float32, [None, layers[0]])
    output_layer    = tf.placeholder(tf.float32, [None, learn_Y.shape[-1]])

    #gamma  = tf.constant tf.Variable(tf.ones([1, learn_Y.shape[-1] - 1]), name='mu')
    l2w = tf.constant(.00001, tf.float32)
    l2b = tf.constant(.0001, tf.float32)
    ns = []

    out = []
    weights = []
    biases = []
    for i in range(1, len(layers)):
        weights.append(tf.Variable(tf.truncated_normal([layers[i-1], layers[i]], stddev=1.)))
        biases.append(tf.Variable(tf.truncated_normal([layers[i]], stddev=.00001)))
        out.append(tf.add(tf.matmul(out[-1] if len(out) > 0 else input_layer, weights[-1]), biases[-1]))

    output = tf.squeeze(tf.tanh(out[-1])) #tf.sub(tf.to_float(tf.argmax(tf.nn.softmax(out[-1]), 1)), tf.constant(2., tf.float32))

    Qt = tf.reduce_sum(tf.mul(tf.reduce_sum(output_layer, 1), output))
    Qm = tf.add(tf.mul(tf.squeeze(tf.slice(output_layer, [0, 0], [nb_batch, 1])), output), tf.reduce_sum(tf.abs(tf.slice(output_layer, [0, 1], [nb_batch, learn_Y.shape[-1] - 1])), 1))
    L = tf.sqrt(tf.reduce_mean(tf.square(Qm - Qt)))
    l2s = tf.constant(0.)
    for i in range(len(weights)):
        l2s += l2w * tf.nn.l2_loss(weights[i]) + l2b * tf.nn.l2_loss(biases[i])
    costf = L + l2s

    # var_list = weights
    # var_list.extend(biases)



    if opt == 'adam':
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-8)
    elif opt == 'adadelta':
        opt = tf.train.AdadeltaOptimizer(lr)

    train_step = opt.minimize(costf)

    result = tf.reduce_sum(tf.reduce_sum(output_layer, 1) * tf.round(output))


    #rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(output_layer, output))))

    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())

    merged_summary_op = tf.merge_all_summaries()
    summary_writer = tf.train.SummaryWriter('/Users/daniilmann/investparty/code/python/27/Futures/logs', graph_def=sess.graph_def)

    e2 = 0
    for i in range(nb_epoch):
        ix = np.random.randint(0, learn_X.shape[0], size=nb_batch)
        lx, ly = learn_X[ix], learn_Y[ix]
        ix = np.random.randint(0, valid_X.shape[0], size=nb_batch)
        vx, vy = valid_X[ix], valid_Y[ix]
        # output.eval(feed_dict={input_layer: lx, output_layer: ly})
        # Qt.eval(feed_dict={input_layer: lx, output_layer: ly})
        # Qm.eval(feed_dict={input_layer: lx, output_layer: ly})
        # e1 = L.eval(feed_dict={input_layer: lx, output_layer: ly})
        # l2s.eval(feed_dict={input_layer: lx, output_layer: ly})
        e2 = .7 * e2 + .3 * costf.eval(feed_dict={input_layer: vx, output_layer: vy})
        train_step.run(feed_dict={input_layer : lx, output_layer : ly})
        # summary_str = merged_summary_op.eval(feed_dict={input_layer : lx, output_layer : ly})
        # summary_writer.add_summary(summary_str, i)
        if i % 100 == 0:
            #print np.mean(weights[-1].eval()), np.std(weights[-1].eval())
            print 'step %s || %.3f || %.3f' % (i, e2, result.eval(feed_dict={input_layer : valid_X, output_layer : valid_Y}))
    print result.eval(feed_dict={input_layer : test_X, output_layer : test_Y})
    res = output.eval(feed_dict={input_layer : test_X, output_layer : test_Y})


    sess.close()

    return res
