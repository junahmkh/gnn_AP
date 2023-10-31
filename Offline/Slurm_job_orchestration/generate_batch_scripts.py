import os

folders = [0,10,12,14,16,18,2,21,23,25,27,29,30,32,34,36,38,40,42,44,46,48,6,8,
           1,11,13,15,17,19,20,22,24,26,28,3,31,33,35,37,39,41,43,45,47,5,7,9]

model = "../Training_and_validation_of_ML_models/GNN_b_scripts"

os.mkdir(model)

for fw in [4,6,12,24,32,64,96,192,288]:
    if((fw==4) or (fw==6) or (fw==12) or (fw==24)):
        t = "02:30:00"
    elif((fw==32) or (fw==64)):
        t = "03:30:00"
    else:
        t = "04:30:00"

    for rack in folders:   
        with open ('../Training_and_validation_of_ML_models/GNN_b_scripts/GNN_{}_{}.sh'.format(fw, rack), 'w') as rsh:
            rsh.write('''\
#!/bin/bash
#SBATCH --account=<FILL IN THE DETAILS ACCORDING TO YOUR SYSTEM>
#SBATCH --partition=<FILL IN THE DETAILS ACCORDING TO YOUR SYSTEM>
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=8G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -c 1
#SBATCH -t {}
python GNN.py {} {}
        '''.format(t,rack, fw))