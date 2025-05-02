'''
打印metric似乎不是必要的
因此文件夹内的maketrain都将返回metric的位置丢弃
'''

from maketrains.mappo_discrete import (
    # make_train as make_train_mappo_discrete,
    save_train as save_train_mappo
)

from maketrains.ppo_discrete import (
    make_train as make_train_ppo_discrete,
)

from maketrains.ppo_discrete_union import (
    make_train_union_vsbaseline as make_train_ppo_discrete_union_vsbaseline,
)
from maketrains.configs import *