import pytest
import tensorflow as tf

from dualing.core import model


def test_base():
    new_base = model.Base(name='new_base')

    assert new_base.name == 'new_base'


def test_base_call():
    new_base = model.Base()

    with pytest.raises(NotImplementedError):
        new_base(None)


def test_siamese():
    new_base = model.Base()

    new_siamese = model.Siamese(new_base)

    assert new_siamese.B == new_base


def test_siamese_setter():
    new_base = model.Base()

    try:
        new_siamese = model.Siamese(None)
    except:
        new_siamese = model.Siamese(new_base)

    assert new_siamese.B == new_base


def test_siamese_compile():
    new_base = model.Base()
    new_siamese = model.Siamese(new_base)

    with pytest.raises(NotImplementedError):
        new_siamese.compile(None)


def test_siamese_step():
    new_base = model.Base()
    new_siamese = model.Siamese(new_base)

    with pytest.raises(NotImplementedError):
        new_siamese.step(None, None)


def test_siamese_fit():
    new_base = model.Base()
    new_siamese = model.Siamese(new_base)

    with pytest.raises(NotImplementedError):
        new_siamese.fit(None)


def test_siamese_evaluate():
    new_base = model.Base()
    new_siamese = model.Siamese(new_base)

    with pytest.raises(NotImplementedError):
        new_siamese.evaluate(None)


def test_siamese_predict():
    new_base = model.Base()
    new_siamese = model.Siamese(new_base)

    with pytest.raises(NotImplementedError):
        new_siamese.predict(None)


def test_siamese_extract_embeddings():
    def call(x):
        return x

    new_base = model.Base()
    new_base.call = call

    new_siamese = model.Siamese(new_base)

    x = tf.zeros(1)

    y = new_siamese.extract_embeddings(x)

    assert y.numpy() == 0
