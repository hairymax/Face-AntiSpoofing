from src.dataset_loader import get_train_loader
from src.config import get_default_config, update_config

conf = get_default_config()

loader = get_train_loader(conf)

data_iter = iter(loader)
sample, ft_sample, target = data_iter.next()

print(target)
print(sample)