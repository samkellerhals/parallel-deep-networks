from models import *
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler
import pyfiglet
import click
from datetime import datetime
import json

trainset = datasets.QMNIST("", train=True, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

testset = datasets.QMNIST("", train=False, download=True, transform=(transforms.Compose([transforms.ToTensor()])))

test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=12, shuffle=True)

def hogwild(model_class, procs, epochs, arch, distributed, nodes, batches):

    torch.set_num_threads(nodes)

    device = torch.device("cpu")
    
    model = model_class.to(device)

    if distributed=='y':

        processes = []

        for rank in range(procs):
            
            #mp.set_start_method('spawn')

            model.share_memory() 

            train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batches, sampler=DistributedSampler(dataset=trainset,num_replicas=procs,rank=rank))

            p = mp.Process(target=train, args=(epochs, arch, model, device, train_loader))

            p.start()

            processes.append(p)

        for p in processes:
            p.join()
        
        test(model, device, test_loader, arch)

    else:
        
        train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=batches, shuffle=True)

        train(epochs, arch, model, device, train_loader)

        test(model, device, test_loader, arch)


def ff_train(arch, epochs, procs, distributed, nodes, batches):
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {procs} processes with distributed processing == {distributed}, {nodes} CPU cores and a batch size of {batches}.')
    
    model_class = FeedforwardNet()
    hogwild(model_class, procs, epochs, arch, distributed, nodes, batches)

def conv_train(arch, epochs, procs, distributed, nodes, batches):    
    click.echo(f'Training neural {arch}-net with {epochs} epochs using {procs} processes with distributed processing == {distributed}, {nodes} CPU cores and a batch size of {batches}.')      
    model_class = ConvNet()
    hogwild(model_class, procs, epochs, arch, distributed, nodes, batches)

@click.command()
@click.option('--epochs', default=1, help='number of epochs to train neural network.')
@click.option('--arch', default='ff', help='neural network architecture to benchmark (conv or ff).')
@click.option('--distributed', default='n', help='whether to distribute data or not (y or n).')
@click.option('--procs', default=1, help='number of processes to spawn.')
@click.option('--nodes', default=1, help='number of cores to use.')
@click.option('--batches', default=12, help='minibatch size to use.')
def main(epochs, arch, procs, distributed, nodes, batches):
    
    intro_text = pyfiglet.figlet_format('Parallel DNN Benchmark', font='slant')
    
    print(intro_text)

    
    
    date_time = datetime.now().strftime("%d%m%Y%H%M%S")

    with open('src/log/' + date_time + '.json', 'w') as f:
        params = {'architecture':arch, 
        'num_epochs':epochs, 
        'num_processes':procs, 
        'threads':nodes, 
        'is_distributed':distributed, 
        'training_time': 'null', 
        'accuracy': 'null',
        'avg_loss': 'null',
        'batch_size': batches
        }

        data = json.dumps(params)
        f.write(data)
        f.close()

    
    if arch == 'ff':            
        ff_train(arch, epochs, procs, distributed, nodes, batches)
    
    elif arch == 'conv':
        conv_train(arch, epochs, procs, distributed, nodes, batches)

    end_text = pyfiglet.figlet_format('Finished benchmark', font='slant')
    
    print(end_text)
    
if __name__ == "__main__":
    main()