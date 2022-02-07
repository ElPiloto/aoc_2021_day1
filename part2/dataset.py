import abc
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf

AOC_INPUT_FILE = './hard_coded/aoc_input.txt'


def label_triplet_pairs(x: np.ndarray):
  """Gets label for pair of triplets."""
  label = np.uint8(np.sum(x[3:]) > np.sum(x[:3]))
  return label



class PairsGenerator(abc.ABC):
  """Generates pairs of numbers with higher or lower label."""
  
  @abc.abstractmethod
  def generator(self):
    pass


class AOCInputFileTripletPairsGenerator(PairsGenerator):
  """Generates pairs data for the AoC input file."""

  def __init__(self, input_file: Optional[str] = AOC_INPUT_FILE, rng_seed: Optional[int] = None):
    self._input_file = input_file
    self.rng_state = np.random.RandomState(rng_seed)
    self._list = np.loadtxt(self._input_file)
    self._num_pairs = len(self._list) - 3
    print(self._num_pairs)
    self._current_idx = 0


  def generator(self):
    def _generator():
      while self._current_idx < self._num_pairs:
        value1 = self._list[self._current_idx:self._current_idx+3]
        value2 = self._list[self._current_idx+1:self._current_idx+4]
        values = np.concatenate([value1, value2])
        values = values.astype(np.float32)
        self._current_idx += 1
        label = label_triplet_pairs(values)
        yield values, label
    return _generator


class TrainingAOCInputFileTripletPairsGenerator(PairsGenerator):
  """Generates in pairs data from AoC input file for training."""

  def __init__(self, only_n_pairs: Optional[int] = -1, input_file: Optional[str] = AOC_INPUT_FILE, rng_seed: Optional[int] = None):
    self._input_file = input_file
    self.rng_state = np.random.RandomState(rng_seed)
    self._list = np.loadtxt(self._input_file)
    self._num_pairs = len(self._list) - 3
    if only_n_pairs > 0:
      # make sure we don't only get the first two values
      self.rng_state.shuffle(self._list)
      self._num_pairs = only_n_pairs
    self._mean = np.mean(self._list)
    self._std = np.std(self._list)
    self._normalize = lambda x: (x - self._mean)/(self._std)
    self._normalize = lambda x: x


  def generator(self):
    def _generator():
      while True:
        current_idx = self.rng_state.randint(0, self._num_pairs - 1)
        value1 = self._list[current_idx:current_idx+3]
        value2 = self._list[current_idx+1:current_idx+4]
        values = np.concatenate([value1, value2])
        values = values.astype(np.float32)
        label = label_triplet_pairs(values)
        yield values, label
    return _generator


class NearbySyntheticTripletPairsGenerator(PairsGenerator):
  """Generates synthetic pairs data with numbers near each other."""

  def __init__(self, min_value: int = 50, max_value: int = 400, rng_seed:
      Optional[int] = None):
    self.rng_state = np.random.RandomState(rng_seed)
    self.min_value = min_value
    self.max_value = max_value

  # TODO(elpiloto): Figure out how to type generators.
  def generator(self):
    def _generator():
      while True:
        values = []
        values.append(self.rng_state.randint(
            low = self.min_value,
            high = self.max_value,
            size=(1)
        ).astype(np.float32))
        for _ in range(1,6):
          offset = self.rng_state.normal(loc=0., scale=10., size=1)
          offset = np.round(offset)
          new_value = values[-1] + offset
          new_value = np.clip(new_value, self.min_value, self.max_value)
          values.append(new_value)
        values = np.array(values)
        values = np.squeeze(values)
        label = label_triplet_pairs(values)
        yield values, label
    return _generator


class BatchDataset:

  def __init__(self, generator):
    self._generator = generator
    # TODO(elpiloto): Add extra config option to normalize each batch
    # by min/max value ever seen.

  def __call__(self, batch_size: int):
    ds = tf.data.Dataset.from_generator(
            self._generator,
            (tf.float32, tf.int32),
            output_shapes=((6,), ()),
    )
    ds = ds.batch(batch_size=batch_size)
    return ds


if False:
  def integration():
    synthetic_generator = SyntheticPairsGenerator(100, 1000)
    ds = BatchDataset(synthetic_generator.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=10).as_numpy_iterator()
  i = 0
  for i in range(10):
    print(next(batch_iterator))
  __import__('pdb').set_trace()


if False:
  def integration():
    aoc_gen = AOCInputFilePairsTripletsGenerator()
    ds = BatchDataset(aoc_gen.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=5).as_numpy_iterator()
  i = 0
  for i in range(10):
    print(next(batch_iterator))

if False:
  def integration():
    synthetic_generator = NearbySyntheticPairsGenerator(100, 1000)
    ds = BatchDataset(synthetic_generator.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=10).as_numpy_iterator()
  i = 0
  for i in range(10):
    print(next(batch_iterator))
  __import__('pdb').set_trace()


if False:
  def integration():
    aoc_gen = TrainingAOCInputFileTripletPairsGenerator()
    ds = BatchDataset(aoc_gen.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=10).as_numpy_iterator()
  i = 0
  for i in range(10):
    print(next(batch_iterator))

if False:
  synthetic_generator = NearbySyntheticPairsGenerator(100, 1000)
  gen = synthetic_generator.generator()
  for (x1, x2), _ in gen():
    print(x2 - x1)


