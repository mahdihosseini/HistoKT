import time

from utils import AverageMeter, get_accuracies, save_model
import sys
import torch
from data import get_data

"""
Imported from Rahavi's MCL code
"""


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
                        optimizer, epoch, config):

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

    return train_loss, test_loss, test_acc


def main():
    config = parse_option()

    # build data loader
    train_loader, test_loader = get_data(name=config.config_yaml["dataset"],
                                         mini_batch_size=,
                                        num_workers: int,
                                        transform_train: transforms,
                                        transform_test: transforms,
                                        level: str = "L3",
                                        dist: bool = False,
                                        transformed_datasets: bool = False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # creating criterion
    dataset_size = len(train_loader.dataset)
    train_class_counts = np.sum(train_loader.dataset.class_labels, axis=0)
    weightsBCE = dataset_size / train_class_counts
    weightsBCE = torch.as_tensor(weightsBCE, dtype=torch.float32).to(device)
    criterion = torch.nn.MultiLabelSoftMarginLoss(weight=weightsBCE).to(device)

    checkpoint_path = "Trained_models/MCLoss_ADP-Release1_resnet18_lr_0.1_decay_0.5_eta_0.0001_step_size_15_level_L3Only/last.pth"
    # checkpoint_path = "Trained_models/Normal/best_transformed.pth"
    model, _, _, _ = load_GMM(checkpoint_path)
    encoder = truncate_GMM(model, encoder_head=True)

    model = FCModel(encoder, 128, 22).to(device)
    # model = FC2layerModel(GMM_model, 512, 22).to(device)

    # build optimizer and scheduler
    optimizer = set_optimizer(config, model)
    scheduler = set_scheduler(config, optimizer)

    # Metrics Calculation
    losses = collections.defaultdict(list)
    losses_filename = "losses_" + \
                      f"date={datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}_" + \
                      f"{config.loss}_" + \
                      f"{config.eta}_" + \
                      f"{config.reg_coeff}" + \
                      f"{config.skew}" + \
                      ".xlsx".replace(' ', '-')

    """___________________Training____________________"""
    for epoch in range(1, config.epochs + 1):

        # train and test for one epoch
        train_loss, test_loss, test_acc = one_epoch_iteration(train_loader, test_loader, model, criterion,
                                                              optimizer, epoch, config)
        scheduler.step()

        # Losses
        losses["train_loss"].append(train_loss)

        losses["test_loss"].append(test_loss)

        losses["test_acc"].append(test_acc)

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

    # save the last model
    save_file = os.path.join(config.save_folder, 'last.pth')
    save_model(model, optimizer, config, config.epochs, save_file)
