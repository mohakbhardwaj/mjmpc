#!/usr/bin/env bash
config_files=('trajopt_reacher-v0_config.yml' 'pen-v0_config') #list of config files
controllers=('random_shooting' 'mppi' 'cem' 'dmd') #list of controllers
run_files=(0) #idxs of config files to run


for ((j=0;j<${#run_files[@]};++j)); do
	config_idx=${run_files[j]}
	printf "====Config file %s ==== \n" ${config_files[config_idx]}  
	# for ((i=0;i<${#run_controllers[@]};++i)); do
	#   controller_idx=${run_controllers[i]}
    python job_script.py --config_file configs/${config_files[config_idx]} --controllers ${controllers[@]} --save_dir ./experiments
	# done
done
