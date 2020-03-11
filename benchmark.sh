#!/bin/bash

# Activate python virtual env
source venv/bin/activate

# Processor count
num_nodes=$( cat /proc/cpuinfo | grep processor -c )

# Non Distributed Training Metrics Generation --1 Process --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net

for (( i = 1; i <= $num_nodes; i++ ))
do  
    echo "feedforward non-distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=n --nodes=$i --arch=ff;
    python src/main.py --epochs=2 --distributed=n --nodes=$i --arch=ff;
    echo "Successfully executed with $i threads."
done

# Convolutional Neural Net

for (( i = 1; i <= $num_nodes; i++ ))
do  
    echo "convolutional non-distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=n --nodes=$i --arch=conv;
    python src/main.py --epochs=2 --distributed=n --nodes=$i --arch=conv;
    echo "Successfully executed with $i threads."
done

# Distributed Training Metrics Generation --2 Processes --Testing CPU increase effect & Arch Difference #

# Feedforward Neural Net

for (( i = 2; i <= $num_nodes; i++ ))
do
    echo "feedforward distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=ff; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=ff;
    echo "Successfully executed with $i threads."
done

# Convolutional Neural Net

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv; 
    echo "Successfully executed with $i threads."
done