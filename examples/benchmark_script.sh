#!/usr/bin/env bash
config_files=('sawyer/reacher-v0.yml' 'sawyer/peg_insertion-v0.yml' 'sawyer/pusher-v0.yml' 'sawyer/door-v0.yml' \
		      'hand/pen-v0.yml' 'hand/door-v0.yml' 'hand/relocate-v0.yml' 'hand/hammer-v0.yml') #list of config files
controllers=('mppi') #list of controllers
run_files=(3) #idxs of config files to run


for ((j=0;j<${#run_files[@]};++j)); do
	config_idx=${run_files[j]}
	printf "====Config file %s ==== \n" ${config_files[config_idx]}  
    python job_script.py --config configs/${config_files[config_idx]} --controllers ${controllers[@]} --save_dir ./experiments
done
