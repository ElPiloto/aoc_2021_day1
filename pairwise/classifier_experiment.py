"""Jaxline experiment to classify greater than or less than pairs."""
import functools
from typing import Dict, Optional

from absl import app
from absl import logging
from absl import flags
import haiku as hk
from haiku import data_structures
import jax
import jax.numpy as jnp
from jaxline import base_config
from jaxline import experiment
from jaxline import platform
from jaxline import utils as jl_utils
import ml_collections
import numpy as np
import optax
from sklearn.metrics import accuracy_score
import tensorflow as tf
import wandb


import dataset


np.set_printoptions(suppress=True, precision=5)
tf.config.set_visible_devices([], 'GPU')
FLAGS = flags.FLAGS

jax.config.update('jax_platform_name', 'gpu')
run = wandb.init(
    project='aoc_2021_day1_pairwise_classifier',
    entity='elpiloto',
)


def get_config():
  # Common config to all jaxline experiments.
  config = base_config.get_base_config()
  config.training_steps = 100000
  config.checkpoint_dir = './checkpoints/'
  # Needed because jaxline version from pypi is broken and version from github
  # breaks everything else.
  config.train_checkpoint_all_hosts = False
  config.interval_type = 'steps'

  # Not common to jaxline
  exp = config.experiment_kwargs = ml_collections.ConfigDict()
  exp.train_seed = 107993
  exp.eval_seed = 8802
  exp.learning_rate = 1e-8
  exp.batch_size = 256
  exp.data_config = ml_collections.ConfigDict()
  train = exp.data_config.train = ml_collections.ConfigDict()
  train.min = 100
  train.max = 10000
  train.only_n_pairs = -1 # only used for 'aoc_train' data
  train.rng_type = 'nearby' # valid values: 'independent', 'aoc_train'

  eval = exp.data_config.eval = ml_collections.ConfigDict()
  eval.name = ["eval"]
  eval.min = [100,]
  eval.max = [10000,]

  model = exp.model = ml_collections.ConfigDict()
  model.output_sizes = [128, 128, 2]
  wandb.config.update(exp.to_dict())
  return config


def cross_entropy_loss(params, model, inputs, targets):
  temperature = 1.
  logits = model.apply(params, inputs)/temperature
  predicted = jnp.argmax(logits, axis=-1)
  # This is Y.
  labels = jax.nn.one_hot(targets, 2)
  cross_entropy = optax.softmax_cross_entropy(logits, labels)
  errors = jnp.abs(predicted - targets)
  return jnp.mean(cross_entropy)
  #return jnp.sum(masked_loss)/jnp.maximum(jnp.sum(errors), 1.)


class Experiment(experiment.AbstractExperiment):

  NON_BROADCAST_CHECKPOINT_ATTRS = {
       '_params': '_params',
       '_opt_state': '_opt_state',
  }

  def __init__(self,
                mode: str,
                train_seed: int,
                eval_seed: int,
                learning_rate: float,
                batch_size: int,
                data_config: ml_collections.ConfigDict,
                model: ml_collections.ConfigDict,
                init_rng: Optional[jnp.DeviceArray] = None):
      super().__init__(mode, init_rng=init_rng)
      self._mode = mode
      self._train_seed = train_seed
      self._eval_seed = eval_seed
      self._learning_rate = learning_rate
      self._data_config = data_config
      self._batch_size = batch_size
      self._config = get_config()
      logging.log(logging.INFO, f'Launched experiment with mode = {mode}')
      run.tags += tuple(FLAGS.wandb_tags)

      # This should really be "train_and_evaluate"
      if mode == 'train':
        self._train_data = self._build_train_data()
        self._eval_datasets = self._build_eval_datasets()
        self._final_eval_data = self._build_final_eval_data()

        train_inputs, _ = next(self._train_data)

        # Initialize model
        model = self._initialize_model()
        self._model = hk.without_apply_rng(hk.transform(model))
        self._params = self._model.init(init_rng, inputs=jnp.zeros_like(train_inputs))

        # We put this in a optax schedule just for easy logging.
        self._sched = sched
        sched = optax.piecewise_constant_schedule(
            self._learning_rate, {10: self._learning_rate}
        )
        opt = optax.adam(learning_rate=sched)
        self._opt_state = opt.init(self._params)
        # Example output, I just like to keep this.
        _ = self._model.apply(self._params, train_inputs)

        # Make update function.
        @jax.jit
        def update_fn(params, inputs, targets):
          loss, grads = jax.value_and_grad(cross_entropy_loss)(params, self._model, inputs,
              targets)
          updates, opt_state = opt.update(grads, self._opt_state, params)
          params = optax.apply_updates(params, updates)
          return params, opt_state, loss
        self._update_fn = jax.jit(update_fn)

      else:
        raise ValueError(f'Unknown mode {mode}')


  def _build_train_data(self):
    """Initializes training data."""
    ds_config = self._data_config['train']
    if ds_config.rng_type == 'nearby':
      generator = dataset.NearbySyntheticPairsGenerator(
          min_value=ds_config.min, max_value=ds_config.max, rng_seed=self._train_seed)
    elif ds_config.rng_type == 'independent':
      generator = dataset.SyntheticPairsGenerator(
          min_value=ds_config.min, max_value=ds_config.max, rng_seed=self._train_seed)
    elif ds_config.rng_type == 'aoc_train':
      generator = dataset.TrainingAOCInputFilePairsGenerator(
          only_n_pairs=ds_config.only_n_pairs,
          rng_seed=self._train_seed)
    print(f'=========Training Data = {type(generator)}=============')
    ds = dataset.BatchDataset(generator.generator())
    batch_iterator = ds(batch_size=self._batch_size).as_numpy_iterator()
    return batch_iterator


  def _build_eval_datasets(self):
    """Initializes eval datasets."""
    ds_config = self._data_config['eval']
    datasets = {}
    for name, min_val, max_val in zip(
        ds_config['name'],
        ds_config['min'],
        ds_config['max'],
        ):
      synthetic_generator = dataset.NearbySyntheticPairsGenerator(
        min_val, max_val, rng_seed=self._eval_seed)
      ds = dataset.BatchDataset(synthetic_generator.generator())
      batch_iterator = ds(batch_size=self._batch_size).as_numpy_iterator()
      datasets[name] = batch_iterator
    return datasets


  def _build_final_eval_data(self):
    """Initializes training data."""
    aoc_input_generator = dataset.AOCInputFilePairsGenerator(rng_seed=self._train_seed)
    ds = dataset.BatchDataset(aoc_input_generator.generator())
    batch_iterator = ds(batch_size=200).as_numpy_iterator()
    return batch_iterator


  def _initialize_model(self):
    """Initializes our model."""
    def forward(inputs):
      output = hk.nets.MLP(
          output_sizes=self._config.experiment_kwargs.model.output_sizes,
          activate_final=False,
          activation=jax.nn.relu,
          )(inputs)
      return output

    return forward

  def step(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:

    # Get next training example
    inputs, targets = next(self._train_data)
    params, opt_state, loss = self._update_fn(self._params, inputs, targets)

    self._params = params
    self._opt_state = opt_state
    learning_rate = self._sched(global_step)[0]
    print(f'Loss: {loss:.6f}, Learning Rate: {learning_rate:.8f}')
    scalars = {'loss': loss, 'learning_rate': learning_rate}
    should_log = False

    if global_step % 50 == 0:
      eval_scalars = self.evaluate(global_step=global_step, rng=rng, writer=writer)
      scalars.update(eval_scalars)
      print(eval_scalars)
      should_log = True

    if global_step % 500 == 0:
    #if global_step == self._config.training_steps - 1:
      final_eval_scalars = self.final_evaluation(global_step=global_step, rng=rng, writer=writer)
      scalars.update(final_eval_scalars)
      should_log = True

    if should_log:
      wandb.log(scalars)

    if writer is not None:
      writer.write_scalars(global_step, scalars)

    return scalars

  def evaluate(self, *, global_step: jnp.ndarray, rng: jnp.ndarray, writer:
      Optional[jl_utils.Writer]) -> Dict[str, np.ndarray]:
    del global_step, rng, writer
    eval_scalars = {}
    for ds_name, eval_data in self._eval_datasets.items():
      inputs, targets = next(eval_data)
      logits = self._model.apply(self._params, inputs)

      predicted = jnp.argmax(logits, axis=-1)

      accuracy = accuracy_score(targets, predicted)
      eval_scalars[f'{ds_name}_accuracy'] = accuracy
    return eval_scalars

  def final_evaluation(self, *, global_step: jnp.ndarray, rng: jnp.ndarray,
      writer: Optional[jl_utils.Writer]) -> Dict[str, int]:
    del global_step, rng, writer
    accuracies = []
    running_count = 0
    i = 0
    for inputs, targets in self._build_final_eval_data():
      logits = self._model.apply(self._params, inputs)
      predicted = jnp.argmax(logits, axis=-1)
      running_count += jnp.sum(predicted)
      accuracy = accuracy_score(targets, predicted)
      accuracies.append(accuracy)
      i += 1
    average_accuracy = np.mean(accuracies)
    param_norms = data_structures.map(lambda _1, _2, x: jnp.linalg.norm(x), self._params)
    print(f'Count: {running_count}, Accuracy: {average_accuracy}, Accuracies: {accuracies}')
    print(f'Final param norms: {param_norms}')
    return {'count': running_count, 'final_eval_accuracy': average_accuracy}



if __name__ == '__main__':
  flags.DEFINE_list('wandb_tags', [], 'Tags to send to wandb.')
  flags.mark_flag_as_required('config')
  app.run(functools.partial(platform.main, Experiment))
