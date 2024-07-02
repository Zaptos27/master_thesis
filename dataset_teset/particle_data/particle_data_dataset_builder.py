"""particle_data dataset."""

import tensorflow_datasets as tfds
import numpy as np
import glob
import re
import tensorflow as tf

class Builder(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for particle_data dataset."""

  VERSION = tfds.core.Version('1.0.1')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(particle_data): Specifies the tfds.core.DatasetInfo object
    return self.dataset_info_from_configs(
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'input': tfds.features.Tensor(shape=(None, 3, 3), dtype=np.float64),
            'target': tfds.features.Tensor(shape=(None, 3), dtype=np.float64)
        }),

        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('input', 'target'),  # Set to `None` to disable
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Returns SplitGenerators."""
    # TODO(particle_data): Downloads the data and defines the splits
    path = "/mnt/HDD/"

    # TODO(particle_data): Returns the Dict[split names, Iterator[Key, Example]]
    return {
        'train': self._generate_examples(path+"train_data/", 'train'),
        'test': self._generate_examples(path+"test_data/", 'test'),
        'val' : self._generate_examples(path+"val_data/", 'val')
    }

  def _generate_examples(self, path, key):
    """Yields examples."""
    input_names = glob.glob(path + "*_input.npy")
    target_names = glob.glob(path + "*_target.npy")
    input_names.sort()
    target_names.sort()
    for i, f in enumerate(input_names):
      print(f, target_names[i])
      input_file = tf.io.gfile.GFile(f, 'rb')
      target_file = tf.io.gfile.GFile(target_names[i], 'rb')
      yield re.findall(r"(\d+)", f)[0], {
          'input': np.load(input_file),
          'target': np.load(target_file),
      }
      input_file.close()
      target_file.close()
