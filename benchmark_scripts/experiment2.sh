#!/bin/bash

# Activate python virtual env
source venv/bin/activate

# Processor countb
num_nodes=$( cat /proc/cpuinfo | grep processor -c )

# Process increase experiment, comparing 2 vs 3 vs 4 processes

# 3 processes conv net

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv; 
    echo "Successfully executed with $i threads."
done

# 4 processes conv net

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv; 
    echo "Successfully executed with $i threads."
done

# Distributed Training Metrics Generation - Minibatch variation experiment - 2 Processes

# 32 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=32; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=32; 
    echo "Successfully executed with $i threads."
done

# 64 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=64; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=64; 
    echo "Successfully executed with $i threads."
done

# 128 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=128; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=128; 
    echo "Successfully executed with $i threads."
done

# 256 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=256; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=2 --arch=conv --batches=256; 
    echo "Successfully executed with $i threads."
done

# Distributed Training Metrics Generation - Minibatch variation experiment - 3 Processes

# 32 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=32; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=32; 
    echo "Successfully executed with $i threads."
done

# 64 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=64; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=64; 
    echo "Successfully executed with $i threads."
done

# 128 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=128; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=128; 
    echo "Successfully executed with $i threads."
done

# 256 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=256; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=3 --arch=conv --batches=256; 
    echo "Successfully executed with $i threads."
done

# Distributed Training Metrics Generation - Minibatch variation experiment - 4 Processes

# 32 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=32; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=32; 
    echo "Successfully executed with $i threads."
done

# 64 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=64; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=64; 
    echo "Successfully executed with $i threads."
done

# 128 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=128; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=128; 
    echo "Successfully executed with $i threads."
done

# 256 minibatch

for (( i = 2; i <= $num_nodes; i++ ))
do  
    echo "convolutional distributed training with $i threads"
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=256; 
    python src/main.py --epochs=2 --distributed=y --nodes=$i --procs=4 --arch=conv --batches=256; 
    echo "Successfully executed with $i threads."
done