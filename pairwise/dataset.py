import abc
from typing import Generator, Optional, Tuple

import numpy as np
import tensorflow as tf

AOC_INPUT_FILE = 'aoc_input.txt'


class PairsGenerator(abc.ABC):
  """Generates pairs of numbers with higher or lower label."""
  
  @abc.abstractmethod
  def generator(self):
    pass


class AOCInputFilePairsGenerator(PairsGenerator):
  """Generates pairs data for the AoC input file."""

  def __init__(self, input_file: Optional[str] = AOC_INPUT_FILE, rng_seed: Optional[int] = None):
    self._input_file = input_file
    self.rng_state = np.random.RandomState(rng_seed)
    self._list = np.loadtxt(self._input_file)
    self._num_pairs = len(self._list) - 1
    self._current_idx = 0


  def generator(self):
    def _generator():
      while self._current_idx < self._num_pairs:
        values = self._list[self._current_idx:self._current_idx+2]
        values = values.astype(np.float32)
        self._current_idx += 1
        label = np.uint8(values[1] > values[0])
        yield values, label
    return _generator



class SyntheticPairsGenerator(PairsGenerator):
  """Generates synthetic pairs data."""

  def __init__(self, min_value: int = 50, max_value: int = 400, rng_seed:
      Optional[int] = None):
    self.rng_state = np.random.RandomState(rng_seed)
    self.min_value = min_value
    self.max_value = max_value

  # TODO(elpiloto): Figure out how to type generators.
  def generator(self):
    def _generator():
      while True:
        values = self.rng_state.randint(
            low = self.min_value, high = self.max_value, size=(2)
        ).astype(np.float32)
        label = np.uint8(values[1] > values[0])
        yield values, label
    return _generator


class NearbySyntheticPairsGenerator(SyntheticPairsGenerator):
  """Generates synthetic pairs data with numbers near each other."""

  # TODO(elpiloto): Figure out how to type generators.
  def generator(self):
    def _generator():
      while True:
        value = self.rng_state.randint(low = self.min_value, high = self.max_value, size=(1)
        ).astype(np.float32)
        offset = np.random.normal(loc=0., scale=10., size=1)
        offset = np.round(offset)
        values = np.array([value, value+offset])
        values = np.clip(values, self.min_value, self.max_value)
        values = np.squeeze(values)
        label = np.uint8(values[1] > values[0])
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
            output_shapes=((2,), ()),
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
    aoc_gen = AOCInputFilePairsGenerator()
    ds = BatchDataset(aoc_gen.generator())
    return ds

  ds = integration()
  batch_iterator = ds(batch_size=200).as_numpy_iterator()
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

