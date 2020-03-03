from models import *
import click

@click.command()
@click.option('--epochs', default=5, help='number of epochs to train neural network.')
@click.option('--arch', default='ff', help='neural network architecture to benchmark.')
def main(epochs, arch):
    if arch == 'ff':
        click.echo(f'Training neural {arch}-net with {epochs} epochs')
        feedforward_model = FeedforwardNet().to(device)
        init_net(epochs, feedforward_model).train()

    elif arch == 'conv':
        click.echo(f'Training neural {arch}-net with {epochs} epochs')
        conv_model = ConvNet().to(device)
        init_net(epochs, conv_model).train()

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