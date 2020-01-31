#!/usr/bin/env bash
config_files=('sawyer_reacher-v0.yml' 'sawyer_peg_insertion-v0.yml' 'sawyer_pusher-v0.yml' 'sawyer_door-v0.yml' \
		      'hand_pen-v0.yml' 'hand_door-v0.yml') #list of config files
controllers=('random_shooting' 'mppi' 'pfmpc') #list of controllers
run_files=(3) #idxs of config files to run


for ((j=0;j<${#run_files[@]};++j)); do
	config_idx=${run_files[j]}
	printf "====Config file %s ==== \n" ${config_files[config_idx]}  
	# for ((i=0;i<${#run_controllers[@]};++i)); do
	#   controller_idx=${run_controllers[i]}
    python job_script.py --config_file configs/${config_files[config_idx]} --controllers ${controllers[@]} --save_dir ./experiments
	# done
done
