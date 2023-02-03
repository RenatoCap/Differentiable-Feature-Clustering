import pytest
import tensorflow as tf
from src.loss import CustomLoss


@pytest.fixture
def loss_fn(kwargs):
    return CustomLoss(**kwargs)


@pytest.mark.parametrize('kwargs', [{'mu': 1, 'q': 1, 'width': 2, 'height': 2}])
def test_continuity_loss_fn(loss_fn):
    test_matrix = tf.ones((1, 2, 2))  # All pixels are the same
    continuity_loss = loss_fn.continuity_loss(test_matrix, test_matrix)
    assert continuity_loss == 0  # Loss should be zero
