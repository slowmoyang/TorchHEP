import dataclasses
from dataclasses import dataclass
from dataclasses import field
from optuna.trial import Trial
from torchhep.utils.config import ConfigBase

def categorical(default, choices):
    if default not in choices:
        raise ValueError
    metadata = {
        'suggest': Trial.suggest_categorical,
        'kwargs': {
            'choices': choices,
        }
    }
    return field(default=default, metadata=metadata)

def boolean(default):
    return categorical(default, [True, False])

def discrete_uniform(default, low, high, q):
    # TODO assert
    metadata = {
        'suggest': Trial.cdiscrete_uniform,
        'kwargs': {
            'low': low,
            'high': high,
            'q': q
        }
    }
    return field(default=default, metadata=metadata)

def floating_point(default, low, high, *, step=None, log=False):
    # TODO assert
    metadata = {
        'suggest': Trial.suggest_float,
        'kwargs': {
            'low': low,
            'high': high,
            'step': step,
            'log': log,
        }
    }
    return field(default=default, metadata=metadata)


def integer(default, low, high, step=1, log=False):
    if not low <= default <= high:
        raise ValueError

    metadata = {
        'suggest': Trial.suggest_int,
        'kwargs': {
            'low': low,
            'high': high,
            'step': step,
            'log': log
        }
    }
    return field(default=default, metadata=metadata)

def loguniform(default, low, high):
    if not low <= default < high:
        raise ValueError
    metadata = {
        'suggest': Trial.suggest_loguniform,
        'kwargs': {
            'low': low,
            'high': high,
        }
    }
    return field(default=default, metadata=metadata)

def uniform(default, low, high):
    if not low <= default < high:
        raise ValueError
    metadata = {
        'suggest': Trial.suggest_uniform,
        'kwargs': {
            'low': low,
            'high': high
        }
    }
    return field(default=default, metadata=metadata)

@dataclass
class HyperparameterConfig(ConfigBase):

    @classmethod
    def from_trial(cls, trial):
        config = {}
        for field in dataclasses.fields(cls):
            if 'suggest' in field.metadata:
                config[field.name] = field.metadata['suggest'](
                    trial, field.name, **field.metadata['kwargs'])
            else:
                if field.default is dataclasses.MISSING:
                    raise ValueError
                config[field.name] = field.default
        return cls(**config)

    @classmethod
    def from_params(cls, params):
        params = cls.from_flat_fields(params) # FIXME
        return cls.from_dict(params)
