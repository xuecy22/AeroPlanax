'''
打印metric似乎不是必要的
因此文件夹内的maketrain都将返回metric的位置丢弃
'''

from maketrains.mappo_discrete import (
    make_train as make_train_mappo_discrete,
    save_train as save_train_mappo
)

from maketrains.mappo_discrete_eval import (
    make_train as make_train_mappo_discrete_eval,
)


from maketrains.ppo_discrete import (
    make_train as make_train_ppo_discrete,
    save_train as save_train_ppo
)

from maketrains.configs import (
    MICRO_CONFIG,
    MINI_CONFIG,
    MEDIUM_CONFIG,
    HUGE_CONFIG
)