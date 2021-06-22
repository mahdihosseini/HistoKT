import time
from datetime import datetime
import yaml

from utils import AverageMeter, get_accuracies, save_model, get_model
import sys
import torch
from data import get_data
import argparse
import os
import numpy as np
from image_transforms import get_transforms

"""
Imported from Rahavi's MCL code
"""

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # Suggested default setting
    parser.add_argument('--save_freq', type=int, default=25,
                        help='save frequency')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')

    # Dataset
    parser.add_argument('--data_folder', type=str, default=None,
                        help='path to the parent containing the dataset folder')

    # Configs
    parser.add_argument('--config', type=str, default="config.yaml",
                        help='path to the config file')

    parser.add_argument('--device_name', type=str, default="",
                        help='device to use')

    config = parser.parse_args()
    config.config_yaml =

    config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
        if not config.device_name else torch.device(config.device_name)

    config.best_acc = 0
    config.start_time = datetime.now()

    # directory for saving the models and tensorboard
    config.tb_folder = os.path.join(config.tb_path, config.model_name)
    if not os.path.isdir(config.tb_folder):
        os.makedirs(config.tb_folder)

    config.save_folder = os.path.join(config.model_path, config.model_name)
    if not os.path.isdir(config.save_folder):
        os.makedirs(config.save_folder)

    if not os.path.isdir(config.eval_folder):
        os.makedirs(config.eval_folder)

    return config

def train(train_loader, model, criterion, optimizer, epoch, config):
    """one epoch training"""
    model.train()

    losses = AverageMeter()
    train_acc = AverageMeter()

    """___________________Training____________________"""

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        batch_size = labels.shape[0]

        # compute loss
        outputs = model(images)
        loss = criterion(outputs, labels)

        # update metric
        losses.update(loss.item(), batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        get_accuracies(outputs, labels, batch_size, train_acc, config)

        # print info
        if (idx + 1) % config.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'
                  'accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                        epoch, idx + 1, len(train_loader),  loss=losses, acc=train_acc))
            sys.stdout.flush()

    return float(losses.avg)


def test(test_loader, model, criterion, config):

    model.eval()

    test_losses = AverageMeter()
    test_acc = AverageMeter()

    with torch.no_grad():
        for idx, (images, labels) in enumerate(test_loader):

            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            batch_size = labels.shape[0]

            outputs = model(images)

            loss = criterion(outputs, labels)
            get_accuracies(outputs, labels, batch_size, test_acc, config)

            # update metric
            test_losses.update(loss.item(), batch_size)

    return float(test_losses.avg), float(test_acc.avg)


def one_epoch_iteration(train_loader, test_loader, model, criterion,
                        optimizer, scheduler, epoch, trial, config):

    start_time = time.time()

    """---------------Training------------------"""
    train_loss = train(train_loader, model, criterion,
                       optimizer, epoch, config)

    """-----------------Testing------------------"""
    test_loss, test_acc = test(test_loader, model, criterion)

    end_time = time.time()

    print('epoch {}\t'
          'Train_loss {:.3f}\t'
          'Test_loss {:.3f} \t'
          'Test_acc {:.3f} \t'
          'total_time {:.2f}'.format(
              epoch, train_loss, test_loss, test_acc, end_time-start_time))
    sys.stdout.flush()

    if epoch % config.save_freq == 0:
        filename = f"trial_{trial}_epoch_{epoch}_date_{config.start_time.strftime('%Y-%m-%d-%H-%M-%S')}.pth"
        save_model(model=model,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   config=config.config_yaml,
                   epoch=epoch,
                   trial=trial,
                   output_filename=config.output_filename,
                   save_file=os.path.join(config.checkpoint_path, filename))
    if np.greater(test_acc, config.best_acc):
        config.best_acc = test_acc
        filename = f"best_trial_{trial}_date_{config.start_time.strftime('%Y-%m-%d-%H-%M-%S')}.pth"
        save_model(model=model,
                   optimizer=optimizer,
                   scheduler=scheduler,
                   config=config.config_yaml,
                   epoch=epoch,
                   trial=trial,
                   output_filename=config.output_filename,
                   save_file=os.path.join(config.checkpoint_path, filename))

    return train_loss, test_loss, test_acc


def main():
    config = parse_option()

    # build data loader

    transform_train, transform_test = get_transforms(
            dataset = config.config_yaml['dataset'],
            degrees = config.config_yaml['degree_of_rotation'],
            gaussian_blur = config.config_yaml['gaussian_blur'],
            kernel_size = config.config_yaml['kernel_size'],
            variance = config.config_yaml['variance'],
            vertical_flipping = config.config_yaml['vertical_flipping'],
            horizontal_flipping = config.config_yaml['horizontal_flipping'],
            horizontal_shift = config.config_yaml['horizontal_shift'],
            vertical_shift = config.config_yaml['vertical_shift'],
            color_kwargs = config.config_yaml['color_kwargs'],
            cutout=config.config_yaml['cutout'],
            n_holes=config.config_yaml['n_holes'],
            length=config.config_yaml['cutout_length'])

    train_loader, train_sampler, test_loader, num_classes = get_data(name=config.config_yaml["dataset"],
            root=config.data_folder,
            mini_batch_size=config.config_yaml['mini_batch_size'],
            num_workers=config.config_yaml['num_workers'],
            transform_train=transform_train,
            transform_test=transform_test,
            dist=False, level=transform_test["level"])

    config.num_classes = num_classes

    # TODO change dist

    # build model and criterion
    model = get_model(config)


    # build optimizer and scheduler
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    # Metrics Calculation
    performance_statistics = dict()
    metrics = Metrics(list(model.parameters()), p=config.p)

    # set the path to save the trained models
    config.model_path = './save/MCLoss/{}_models'.format(config.dataset)
    config.output_filename = "results_" + \
                             f"date={self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}_" + \
                             f"trial={trial}_" + \
                             f"{self.config['network']}_" + \
                             f"{self.config['dataset']}_" + \
                             f"{self.config['optimizer']}" + \
                             '_'.join([f"{k}={v}" for k, v in
                                       self.config['optimizer_kwargs'].items()]) + \
                             f"_{self.config['scheduler']}" + \
                             '_'.join([f"{k}={v}" for k, v in
                                       self.config['scheduler_kwargs'].items()]) + \
                             f"_LR={learning_rate}" + \
                             ".xlsx".replace(' ', '-')
    self.output_filename = str(
        lr_output_path / self.output_filename)

    """___________________Training____________________"""
    for epoch in range(1, config.epochs + 1):

        # train and test for one epoch
        train_loss, test_loss = one_epoch_iteration(train_loader, test_loader, model, criterion,
                                                    optimizer, epoch, config, history, logger)
        if scheduler:
            scheduler.step()

        # Losses
        losses["train_NLLLoss"].append(train_loss[0])
        losses["train_GJSDLoss"].append(train_loss[1])
        losses["train_loss"].append(train_loss[2])

        losses["test_NLLLoss"].append(test_loss[0])
        losses["test_GJSDLoss"].append(test_loss[1])
        losses["test_loss"].append(test_loss[2])

        # Performance metrics
        io_metrics = metrics.evaluate(epoch - 1)
        performance_statistics[f'in_S_epoch_{epoch}'] = io_metrics.input_channel_S
        performance_statistics[f'out_S_epoch_{epoch}'] = io_metrics.output_channel_S
        performance_statistics[f'fc_S_epoch_{epoch}'] = io_metrics.fc_S
        performance_statistics[f'in_rank_epoch_{epoch}'] = io_metrics.input_channel_rank
        performance_statistics[f'out_rank_epoch_{epoch}'] = io_metrics.output_channel_rank
        performance_statistics[f'fc_rank_epoch_{epoch}'] = io_metrics.fc_rank
        performance_statistics[f'in_condition_epoch_{epoch}'] = io_metrics.input_channel_condition
        performance_statistics[f'out_condition_epoch_{epoch}'] = io_metrics.output_channel_condition
        performance_statistics[f'learning_rate_epoch_{epoch}'] = optimizer.param_groups[0]['lr']

        # Store the performance metrics in a dataframe
        df = pd.DataFrame(data=performance_statistics)
        # Save the dataframe to a excel sheet
        df.to_excel(os.path.join(config.eval_folder, output_filename))

        # save the model
        if epoch % config.save_freq == 0:
            save_file = os.path.join(
                config.save_folder, 'checkpoints_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, config, epoch, save_file)

    # Store the losses in a dataframe
    loss_df = pd.DataFrame(data=losses)
    # Save the loss dataframe in a excel
    loss_df.to_excel(os.path.join(config.eval_folder, losses_filename))
    # Plot the losses
    plot_loss_df(loss_df, config)

    filename = f"last_trial_{trial}_date_{self.start_time.strftime('%Y-%m-%d-%H-%M-%S')}.pth.tar"
    torch.save(data, str(self.checkpoint_path / filename))
    # resetting best acc for each trial
    self.best_acc1 = 0

    # save the last model
    save_file = os.path.join(config.save_folder, 'last.pth')
    save_model(model, optimizer, config, config.epochs, save_file)
