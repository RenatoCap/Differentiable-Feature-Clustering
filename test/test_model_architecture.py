import pytest
import tensorflow as tf

from src.diff_feature_model import DiffFeatureModel

params_list = [{'img_shape': (10, 10), 'p': 1, 'q': 1, 'm': 1},
               {'img_shape': (100, 100), 'p': 5, 'q': 5, 'm': 5},
               {'img_shape': (224, 224), 'p': 10, 'q': 10, 'm': 10}]


# This model function can be passed as parameter for use in test functions
@pytest.fixture
def model(kwargs):
    return DiffFeatureModel(**kwargs)


@pytest.mark.parametrize('kwargs', params_list)
def test_model_working_correctly(model):
    """
    Verifies that the model can be used for inference and gives the correct output shape
    """
    input_test = tf.random.normal((1, *model.shape, 3))
    response = model(input_test)
    assert response.shape == (1, *model.shape, model.M)
