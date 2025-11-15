


from Datasets.base_dataset import BaseDataset
from Datasets.dataset_dota import DOTA
from Datasets.dataset_hrsc import HRSC
from Datasets.dataset_fair1m import FAIR1M

__all__ = ['build_dataset']
support_dataset = ['BaseDataset', 'HRSC', 'DOTA', 'FAIR1M']


def build_dataset(phase, config):
    """
    phase:状态,train
    """
    assert config['type'] in support_dataset, (f'{config["type"]} is not developed yet!, '
                                               f'only {support_dataset} are support now')
    arch_model = eval(config['type'])(phase, config)
    return arch_model
