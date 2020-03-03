from models import *
import torch.multiprocessing as mp
import click

def hogwild():
    pass

def ff_train(arch, epochs, procs, workers):
    click.echo(f'Training neural {arch}-net with {epochs} epochs on {device}')
    torch.set_num_threads(workers)
    mp.set_start_method('spawn')
    feedforward_model = FeedforwardNet().to(device)
    feedforward_model.share_memory() # gradients are allocated lazily, so they are not shared here
    processes = []

    for rank in range(procs):
        p = mp.Process(target=init_net(epochs, feedforward_model).train())
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

def conv_train(arch, epochs):
    click.echo(f'Training neural {arch}-net with {epochs} epochs on {device}')
    feedforward_model = FeedforwardNet().to(device)
    init_net(epochs, feedforward_model).train()

@click.command()
@click.option('--epochs', default=5, help='number of epochs to train neural network.')
@click.option('--arch', default='ff', help='neural network architecture to benchmark.')
@click.option('--procs', default=2, help='number of processes to spawn')
@click.option('--workers', default=2, help='number of cores to use')
def main(epochs, arch, workers, procs):
    if arch == 'ff':
        ff_train(arch, epochs, workers, procs)
        

    elif arch == 'conv':
        conv_train(arch, epochs)

if __name__ == "__main__":
    main()

'''
TODO:
Options
- batch size
- where to log results

Flags
- GPU or CPU
- Number Workers

App screen
- use figlet for cmd font
'''