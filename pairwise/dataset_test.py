from absl.testing import absltest
import numpy as np

import dataset


class PairsGeneratorTest(absltest.TestCase):

  def _batch_check_label_direction(self, batch_inputs, batch_targets):
    """Checks label = 1 if greater, 0 if not."""
    batch_size = batch_inputs.shape[0]
    for b in range(batch_size):
      if batch_inputs[b, 1] > batch_inputs[b, 0]:
        self.assertEqual(batch_targets[b], 1,
            f'Second number is greater than first, should output 1, not 0.\nInputs: {batch_inputs[b]}, Target: {batch_targets[b]}.')
      else:
        self.assertEqual(batch_targets[b], 0,
            f'Second number is not greater than first, should output 0, not 1.\nInputs: {batch_inputs[b]}, Target: {batch_targets[b]}.')


class AOCInputFilePairsGeneratorTests(PairsGeneratorTest):

  def setUp(self):
    aoc_gen = dataset.AOCInputFilePairsGenerator()
    ds = dataset.BatchDataset(aoc_gen.generator())
    self._batch_iterator = ds(batch_size=200).as_numpy_iterator()

  def test_range(self):
    min = np.inf
    max = -np.inf
    for inputs, _ in self._batch_iterator:
      min = np.min([min, np.min(inputs)])
      max = np.max([max, np.max(inputs)])
    self.assertEqual(min, 100., f'Expect min to be 100, but acutally: {min}')
    self.assertEqual(max, 10044., f'Expect max to be 10044, but acutally: {max}')

  def test_label(self):
    """Checks label = 1 if greater, 0 if not."""
    inputs, targets = next(self._batch_iterator)
    self._batch_check_label_direction(inputs, targets)


class SynthethicPairsGeneratorTests(PairsGeneratorTest):

  def setUp(self):
    synth_gen = dataset.SyntheticPairsGenerator()
    ds = dataset.BatchDataset(synth_gen.generator())
    self._batch_iterator = ds(batch_size=200).as_numpy_iterator()

  def test_label(self):
    """Checks label = 1 if greater, 0 if not."""
    inputs, targets = next(self._batch_iterator)
    self._batch_check_label_direction(inputs, targets)


class NearbySynthethicPairsGeneratorTests(PairsGeneratorTest):

  def setUp(self):
    synth_gen = dataset.NearbySyntheticPairsGenerator()
    ds = dataset.BatchDataset(synth_gen.generator())
    self._batch_iterator = ds(batch_size=200).as_numpy_iterator()

  def test_label(self):
    """Checks label = 1 if greater, 0 if not."""
    inputs, targets = next(self._batch_iterator)
    self._batch_check_label_direction(inputs, targets)

if __name__ == '__main__':
  absltest.main()
