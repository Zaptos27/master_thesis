"""particle_data dataset."""

from . import particle_data_dataset_builder
import tensorflow_datasets as tfds

class ParticleDataTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for particle_data dataset."""
  # TODO(particle_data):
  DATASET_CLASS = particle_data_dataset_builder.Builder
  SPLITS = {
      'train': 10,  # Number of fake train example
      'test': 3,  # Number of fake test example
      'val': 3,  # Number of fake val example
  }

  # If you are calling `download/download_and_extract` with a dict, like:
  #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
  # then the tests needs to provide the fake output paths relative to the
  # fake data directory
  # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == '__main__':
  tfds.testing.test_main()
