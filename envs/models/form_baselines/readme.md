|name||succcess_rate|
|:--:|:--:|:--:|
|form_0415_cp440|cube_a=555, success_d=200|99%|
|form_0415_cp520|cube_a=555, success_d=100|97%|
|form_0415_cp560|cube_a=555, success_d=50|99%|
|form_0415_cp920|cube_a=555, init_yaw=(-pi, pi), success_d=50|95%|
|form_0420_cp960|communicate_range=50000.0|70%+?但用于render似乎更高，可能只是增加了episode_length，或者在飞机距离较近时容易出问题。|


从200m->50m的过程中，似乎不需要人为给yaw增加reward了  
适用于原始版本网络（obs 15ego+5other）