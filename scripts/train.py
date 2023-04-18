import argparse
import pathlib
import time 

from tqdm import tqdm
import numpy as np 
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import SemIRLDataset
from model import SemIRLModel


def train(model, trainloader, criterion, optimizer):
    model = model.train()

    total_loss = []
    for i, data in enumerate(trainloader):

        grid_count = data[0].float().to(device='cuda')
        agent_pos = data[1].to(device='cuda')
        goal_pos = data[2].to(device='cuda')
        action = data[3].to(device='cuda')

        logit, policy = model(grid_count, agent_pos, goal_pos)

        loss = criterion(logit, action)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss.append(loss.item())
    return np.mean(total_loss)

def validate(model, validloader, criterion):
    model = model.eval()

    total_loss = []
    total_acc = []
    for i, data in enumerate(validloader):
        grid_count = data[0].to(device="cuda")
        agent_pos = data[1].to(device="cuda")
        goal_pos = data[2].to(device="cuda")
        action = data[3].to(device="cuda")

        # Forward pass
        logit, policy = model(grid_count, agent_pos, goal_pos)

        # Calculate Loss and Error
        loss = criterion(logit, action)

        total_loss.append(loss.item())
        cp = np.argmax(policy.cpu().detach().numpy(), 1)
        acc = np.mean(cp == action.cpu().detach().numpy())
        total_acc.append(acc)

    metrics = {'loss': np.mean(total_loss), 'acc': np.mean(total_acc)}
    return metrics

def save_model(model, directory):
    torch.save(model.state_dict(), f"{directory}/model.pt")

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size', type=int, default=16, const=16, nargs='?',
        choices=[16, 64, 128], help='Minigrid map size')
    parser.add_argument('--batch_size', type=int, default=128, help="Minibatch size for training")
    parser.add_argument('--weight_decay', type=float, default=0.001, help="L2 regularization in optimizer")
    args = parser.parse_args()


    log_dir = pathlib.Path('./logs') / time.strftime("%m.%d.%Y") / time.strftime("%H-%M-%S")
    trained_model_dir = log_dir / 'trained_model'
    pathlib.Path(trained_model_dir).mkdir(parents=True, exist_ok=False)

    directory = pathlib.Path(f'./demonstrations/MiniGrid-LavaLawnS{args.grid_size}-v0')
    print(f"Loading dataset from {directory}")
    trainset = SemIRLDataset(directory / 'train')
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    validset = SemIRLDataset(directory / 'valid')
    validloader = DataLoader(validset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = SemIRLModel(grid_size=args.grid_size, batch_size=args.batch_size).cuda()
    weight_decay = args.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)

    curr_best_acc = 0
    for epoch in tqdm(range(1000)):
        loss = train(model, trainloader, criterion, optimizer)
        writer.add_scalar("train/loss", loss, epoch)

        valid_metrics = validate(model, validloader, criterion)
        for k, v in valid_metrics.items():
            writer.add_scalar(f"valid/{k}", v, epoch)
        writer.add_scalar("weight_decay", weight_decay, epoch)
        writer.flush()

        # Save models
        if curr_best_acc < valid_metrics['acc']:
            curr_best_acc = valid_metrics['acc']
            print(f"Current best acc: {curr_best_acc:.2f}, at epoch {epoch}")

            directory = trained_model_dir / 'best_model'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
            save_model(model, directory)    
            
        if epoch % 100 == 0:
            directory = trained_model_dir / f'epoch_{epoch:04d}'
            pathlib.Path(directory).mkdir(parents=True, exist_ok=False)
            save_model(model, directory)

    writer.close()

if __name__ == '__main__':
    main()