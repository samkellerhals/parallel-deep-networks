from models import *
import click

@click.command()
@click.option('--epochs', default=5, help='number of epochs to train neural network.')
def main(epochs):
    click.echo(f'Training neural net with {epochs} epochs')
    x = train_feedforward(epochs)
    x.train()

if __name__ == "__main__":
    main()

'''
Make arguments for CLI
- type of DNN to train

Add Options
- batch size
- where to log results

Flags (enable/disable certain behavior)
-GPU/No GPU

use figlet for cmd font
use click

'''