# coding=utf-8
# Copyright 2020 The Paper Authors. All Rights Reserved.
#
# You may not modify, distribute, or use this software
# for any purpose beyond the NeurIPS 2020 review process.
#
# =======================================================================
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import functools
import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from tqdm import trange

from cta.cta_remixmatch import CTAReMixMatch
from libml import data, utils, augment, ctaugment
from libml.gradient import gradients as custom_gradient

FLAGS = flags.FLAGS


class AugmentPoolCTACutOut(augment.AugmentPoolCTA):
    @staticmethod
    def numpy_apply_policies(arglist):
        x, cta, probe = arglist
        if x.ndim == 3:
            assert probe
            policy = cta.policy(probe=True)
            return dict(policy=policy,
                        probe=ctaugment.apply(x, policy),
                        image=x)
        assert not probe
        cutout_policy = lambda: cta.policy(probe=False) + [ctaugment.OP('cutout', (1,))]
        return dict(image=np.stack([x[0]] + [ctaugment.apply(y, cutout_policy()) for y in x[1:]]).astype('f'))


class FixMatch(CTAReMixMatch):
    AUGMENT_POOL_CLASS = AugmentPoolCTACutOut

    def __init__(self, train_dir: str, dataset: data.DataSets, nclass: int, **kwargs):
        super().__init__(train_dir, dataset, nclass, **kwargs)
        self.classify_v = self.model_val(**kwargs)
        self.model_per_ex = self.model_per_ex(nclass=nclass, **kwargs)
        self.weights = np.ones(kwargs['size_unlabeled'], dtype=np.float32)
        self.m_ts = np.zeros(kwargs['size_unlabeled'], dtype=np.float32)
        self.v_ts = np.zeros(kwargs['size_unlabeled'], dtype=np.float32)
        self.ts = np.zeros(kwargs['size_unlabeled'], dtype=np.float32)
        self.alpha = kwargs['alpha']
        self.inf_warm = kwargs['inf_warm']
        self.inner_steps = kwargs['inner_steps']

    def compute_influence(self, grad_vec_wrt_val_loss, x, y, w_match):
        inv_H = self.session.run(self.ops.inv_H_op,
                                 feed_dict={self.ops.y: y['image'],
                                            self.ops.xt: x['image'],
                                            self.ops.label: x['label'],
                                            self.ops.w_match: w_match})
        grads_train_per_ex = self.session.run(self.model_per_ex.grads_train_per_ex,
                                              feed_dict={self.model_per_ex.y: y['image'],
                                                         self.model_per_ex.w_match: w_match})
        jacobian_per_ex_u = grads_train_per_ex['classify/dense/kernel/grad'][FLAGS.batch * FLAGS.uratio:]
        jacobian_per_ex_u = jacobian_per_ex_u.reshape(
            FLAGS.batch * FLAGS.uratio, -1).transpose()
        grad_vec_wrt_val = grad_vec_wrt_val_loss.reshape(1, -1)
        influences = -np.matmul(np.matmul(grad_vec_wrt_val, inv_H), jacobian_per_ex_u)[0]
        return influences

    def train_step(self, train_session, gen_labeled, gen_unlabeled, ep=0):
        x, y = gen_labeled(), gen_unlabeled()
        batch_ids = y['index'][:, 0]
        w_match = self.weights[batch_ids]

        # update per-example weights
        if ep > self.inf_warm and self.tmp.step % (self.inner_steps * FLAGS.batch) == 0:
            grad_vec_wrt_val_loss = self.grad_wrt_val_loss_batch()
            influences = self.compute_influence(
                grad_vec_wrt_val_loss, x, y, w_match)
            self.madam_update_influence(batch_ids, influences)

        # train the network use the updated influence as weight
        self.tmp.step = train_session.run([self.ops.train_op, self.ops.update_step],
                                          feed_dict={self.ops.y: y['image'],
                                                     self.ops.xt: x['image'],
                                                     self.ops.label: x['label'],
                                                     self.ops.w_match: w_match})[1]

    def get_all_params(self):
        all_params = []
        for v in utils.model_vars(None):
            temp_tensor = tf.get_default_graph().get_tensor_by_name(v.name)
            all_params.append(temp_tensor)
        return all_params

    def grad_wrt_val_loss_batch(self, batch=None, feed_extra=None, **kwargs):
        """ get grad of loss on validation set w.r.t the params """
        batch = batch or FLAGS.batch
        images, labels = self.tmp.cache['valid']
        indexes = np.random.randint(
            low=0, high=images.shape[0], size=batch * self.params['uratio'])
        grad = self.session.run(
            self.classify_v.grad_val_loss_op,
            feed_dict={
                self.classify_v.v: images[indexes],
                self.classify_v.l: labels[indexes],
                **(feed_extra or {})
            })
        grad_vec = np.concatenate([g.flatten() for g in grad])
        # print('Total number of parameters: %s' % len(grad_vec))
        return grad_vec

    def train(self, train_nimg, report_nimg):
        if FLAGS.eval_ckpt:
            self.eval_checkpoint(FLAGS.eval_ckpt)
            return

        batch = FLAGS.batch
        train_labeled = self.dataset.train_labeled.repeat().shuffle(
            FLAGS.shuffle).parse().augment()
        train_labeled = train_labeled.batch(batch).prefetch(
            16).make_one_shot_iterator().get_next()
        train_unlabeled = self.dataset.train_unlabeled.repeat().shuffle(
            FLAGS.shuffle).parse().augment()
        train_unlabeled = train_unlabeled.batch(
            batch * self.params['uratio']).prefetch(16)
        train_unlabeled = train_unlabeled.make_one_shot_iterator().get_next()
        scaffold = tf.train.Scaffold(saver=tf.train.Saver(max_to_keep=FLAGS.keep_ckpt,
                                                          pad_step_number=10))

        with tf.Session(config=utils.get_config()) as sess:
            self.session = sess
            self.cache_eval()

        with tf.train.MonitoredTrainingSession(
                scaffold=scaffold,
                checkpoint_dir=self.checkpoint_dir,
                config=utils.get_config(),
                save_checkpoint_steps=FLAGS.save_kimg << 10,
                save_summaries_steps=report_nimg - batch) as train_session:
            self.session = train_session._tf_sess()
            gen_labeled = self.gen_labeled_fn(train_labeled)
            gen_unlabeled = self.gen_unlabeled_fn(train_unlabeled)
            self.tmp.step = self.session.run(self.step)
            while self.tmp.step < train_nimg:
                loop = trange(self.tmp.step % report_nimg, report_nimg, batch,
                              leave=False, unit='img', unit_scale=batch,
                              desc='Epoch %d/%d' % (1 + (self.tmp.step // report_nimg), train_nimg // report_nimg))
                ep = 1 + (self.tmp.step // report_nimg)
                for _ in loop:
                    self.train_step(train_session, gen_labeled,
                                    gen_unlabeled, ep)
                    while self.tmp.print_queue:
                        loop.write(self.tmp.print_queue.pop(0))
            while self.tmp.print_queue:
                print(self.tmp.print_queue.pop(0))

    def model(self, batch, lr, wd, wu, confidence, uratio, ema=0.999, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        xt_in = tf.placeholder(
            tf.float32, [batch] + hwc, 'xt')  # Training labeled
        x_in = tf.placeholder(tf.float32, [None] + hwc, 'x')  # Eval images
        # Training unlabeled (weak, strong)
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')
        l_in = tf.placeholder(tf.int32, [batch], 'labels')  # Labels
        # weights for unlabeled data
        w_match = tf.placeholder(tf.float32, [batch * uratio], 'w_match')

        lrate = tf.clip_by_value(tf.to_float(
            self.step) / (FLAGS.train_kimg << 10), 0, 1)
        lr *= tf.cos(lrate * (7 * np.pi) / (2 * 8))
        tf.summary.scalar('monitors/lr', lr)

        # Compute logits for xt_in and y_in
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        x = utils.interleave(
            tf.concat([xt_in, y_in[:, 0], y_in[:, 1]], 0), 2 * uratio + 1)
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        logits = utils.de_interleave(logits, 2 * uratio + 1)
        post_ops = [v for v in tf.get_collection(
            tf.GraphKeys.UPDATE_OPS) if v not in skip_ops]
        logits_x = logits[:batch]
        logits_weak, logits_strong = tf.split(logits[batch:], 2)
        del logits, skip_ops

        # Labeled cross-entropy
        l_in_1hot = tf.one_hot(l_in, kwargs['nclass'])
        loss_xe = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=l_in_1hot, logits=logits_x)
        loss_xe = tf.reduce_mean(loss_xe)
        tf.summary.scalar('losses/xe', loss_xe)

        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        pseudo_labels_hard = tf.one_hot(
            tf.argmax(pseudo_labels, axis=1), kwargs['nclass'])
        loss_xeu_all = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=pseudo_labels_hard, logits=logits_strong)
        pseudo_mask = tf.to_float(tf.reduce_max(
            pseudo_labels, axis=1) >= confidence)
        tf.summary.scalar('monitors/mask', tf.reduce_mean(pseudo_mask))
        tf.summary.scalar('monitors/lambdas', tf.reduce_mean(w_match))
        loss_xeu = tf.reduce_mean(loss_xeu_all * pseudo_mask * w_match)
        tf.summary.scalar('losses/xeu', loss_xeu)

        # L2 regularization
        loss_wd = sum(tf.nn.l2_loss(v)
                      for v in utils.model_vars('classify') if 'kernel' in v.name)
        tf.summary.scalar('losses/wd', loss_wd)

        # inverse hessian
        self.model_params = self.get_all_params()[-2]
        total_loss = loss_wd + loss_xeu + loss_xe
        hessian = tf.hessians(total_loss, self.model_params)
        # TODO: remove the hard-coded 128
        _dim = 128 * kwargs['nclass']
        hessian = tf.reshape(hessian, [_dim, _dim])
        inv_H_op = tf.linalg.inv(hessian)

        ema = tf.train.ExponentialMovingAverage(decay=ema)
        ema_op = ema.apply(utils.model_vars())
        ema_getter = functools.partial(utils.getter_ema, ema)
        post_ops.append(ema_op)

        train_op = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(
            loss_xe + wu * loss_xeu + wd * loss_wd, colocate_gradients_with_ops=True)
        with tf.control_dependencies([train_op]):
            train_op = tf.group(*post_ops)

        return utils.EasyDict(
            xt=xt_in, x=x_in, y=y_in, label=l_in, train_op=train_op,
            w_match=w_match, inv_H_op=inv_H_op,
            classify_raw=tf.nn.softmax(classifier(x_in, training=False)),
            classify_op=tf.nn.softmax(classifier(x_in, getter=ema_getter, training=False)))

    def model_val(self, batch, uratio, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        # validation labeled
        x = tf.placeholder(tf.float32, [batch * uratio] + hwc, 'v')
        l_in = tf.placeholder(tf.int32, [batch * uratio], 'labels')  # Labels
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        logits = utils.para_cat(lambda x: classifier(x, training=True), x)
        loss_xe = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=l_in, logits=logits))  # loss
        grad_val_loss_op = tf.gradients(loss_xe, self.model_params)
        return utils.EasyDict(v=x, l=l_in, grad_val_loss_op=grad_val_loss_op)

    def model_per_ex(self, nclass, batch, confidence, uratio, **kwargs):
        hwc = [self.dataset.height, self.dataset.width, self.dataset.colors]
        y_in = tf.placeholder(tf.float32, [batch * uratio, 2] + hwc, 'y')
        # weights for unlabeled data
        w_match = tf.placeholder(tf.float32, [batch * uratio], 'w_match')
        # forward
        classifier = lambda x, **kw: self.classifier(x, **kw, **kwargs).logits
        x = tf.concat([y_in[:, 0], y_in[:, 1]], 0)
        logits = classifier(x, training=True)
        logits_weak, logits_strong = tf.split(logits, 2)
        # Pseudo-label cross entropy for unlabeled data
        pseudo_labels = tf.stop_gradient(tf.nn.softmax(logits_weak))
        pseudo_labels_hard = tf.one_hot(
            tf.argmax(pseudo_labels, axis=1), nclass)
        loss_xeu_all = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=pseudo_labels_hard, logits=logits_strong)
        pseudo_mask = tf.to_float(tf.reduce_max(
            pseudo_labels, axis=1) >= confidence)
        loss_xeu = tf.reduce_mean(loss_xeu_all * pseudo_mask * w_match)
        # per-ex-grad_wrt_unlabeled_loss
        grads_train_per_ex = custom_gradient(loss_xeu_all, self.model_params)
        return utils.EasyDict(y=y_in, w_match=w_match, grads_train_per_ex=grads_train_per_ex)

    def madam_update_influence(self, batch_ids, grad, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        if not isinstance(grad, np.ndarray):
            grad = np.array(grad)
        weight = self.weights[batch_ids]
        m_t = self.m_ts[batch_ids]
        v_t = self.v_ts[batch_ids]
        t = self.ts[batch_ids]

        t += 1
        # updates the moving averages of the gradient
        m_t = beta_1 * m_t + (1 - beta_1) * grad
        # updates the moving averages of the squared gradient
        v_t = beta_2 * v_t + (1 - beta_2) * (grad * grad)
        # calculates the bias-corrected estimates
        m_cap = m_t / (1 - (beta_1**t))
        # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta_2**t))

        weight -= (self.alpha * m_cap) / (np.sqrt(v_cap) + epsilon)
        weight[weight < 0] = 0
        weight[weight > 2] = 2

        self.weights[batch_ids] = weight
        self.m_ts[batch_ids] = m_t
        self.v_ts[batch_ids] = v_t
        self.ts[batch_ids] = t


def main(argv):
    utils.setup_main()
    del argv  # Unused.
    dataset = data.PAIR_DATASETS()[FLAGS.dataset]()
    log_width = utils.ilog2(dataset.width)
    model = FixMatch(
        os.path.join(FLAGS.train_dir, dataset.name, FixMatch.cta_name()),
        dataset,
        lr=FLAGS.lr,
        wd=FLAGS.wd,
        arch=FLAGS.arch,
        batch=FLAGS.batch,
        nclass=dataset.nclass,
        wu=FLAGS.wu,
        confidence=FLAGS.confidence,
        uratio=FLAGS.uratio,
        scales=FLAGS.scales or (log_width - 2),
        filters=FLAGS.filters,
        repeat=FLAGS.repeat,
        size_unlabeled=dataset.size_unlabeled,
        alpha=FLAGS.alpha,
        inf_warm=FLAGS.inf_warm,
        inner_steps=FLAGS.inner_steps,
    )
    model.train(FLAGS.train_kimg << 10, FLAGS.report_kimg << 10)


if __name__ == '__main__':
    utils.setup_tf()
    flags.DEFINE_float('confidence', 0.95, 'Confidence threshold.')
    flags.DEFINE_float('wd', 0.0005, 'Weight decay.')
    flags.DEFINE_float('wu', 1, 'Pseudo label loss weight.')
    flags.DEFINE_integer('filters', 32, 'Filter size of convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of residual layers per stage.')
    flags.DEFINE_integer(
        'scales', 0, 'Number of 2x2 downscalings in the classifier.')
    flags.DEFINE_integer('uratio', 7, 'Unlabeled batch size ratio.')
    FLAGS.set_default('augment', 'd.d.d')
    FLAGS.set_default('dataset', 'cifar10.3@250-1')
    FLAGS.set_default('batch', 64)
    FLAGS.set_default('lr', 0.03)
    FLAGS.set_default('train_kimg', 1 << 16)

    flags.DEFINE_float('alpha', 0.01, 'learning rate to update lamnda.')
    flags.DEFINE_integer('inf_warm', 0, 'influence computing warmp-up')
    flags.DEFINE_integer('inner_steps', 100, 'how often to update lambdas')

    app.run(main)
