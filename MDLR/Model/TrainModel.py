import torch
import time
import os

from Model.clincial_dataset import get_readData
from Model.LinePlot import excelplot_acc
from Model.focal_loss import FocalLoss



class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.acc = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_epoch(model, loader, optimizer, epoch, n_epochs, print_freq=1):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    acc = AverageMeter()

    # Model on train mode
    model.train()
    end = time.time()

    lossfuction = FocalLoss(alpha=0.75, gamma=2, num_classes=2)
    for batch_idx, (input, target,PathName) in enumerate(loader):
        # Create vaiables
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = torch.nn.functional.cross_entropy(output, target)
        #loss = lossfuction(output, target)


        # measure accuracy and record loss
        batch_size = target.size(0)
        _, pred = output.data.cpu().topk(1, dim=1)
        error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        acc.update(torch.eq(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
        losses.update(loss.item(), batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print stats
        if batch_idx % print_freq == 0:
            res = '\t'.join([
                'Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                'Error %.4f (%.4f)' % (error.val*100, error.avg*100),
                'Acc %.4f (%.4f)' % (acc.val*100,acc.avg*100),
            ])
            print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg


def eval_epoch(model, loader, print_freq=1, is_test=True):
    batch_time = AverageMeter()
    losses = AverageMeter()
    error = AverageMeter()
    acc = AverageMeter()

    lossfuction = FocalLoss(alpha=0.75, gamma=2, num_classes=2)
    # Model on eval mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for batch_idx, (input, target,PathName) in enumerate(loader):
            # Create vaiables
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = torch.nn.functional.cross_entropy(output, target)
            #loss = lossfuction(output, target)

            # measure accuracy and record loss
            batch_size = target.size(0)
            _, pred = output.data.cpu().topk(1, dim=1)
            error.update(torch.ne(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            acc.update(torch.eq(pred.squeeze(), target.cpu()).float().sum().item() / batch_size, batch_size)
            losses.update(loss.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print stats
            if batch_idx % print_freq == 0:
                res = '\t'.join([
                    'Test' if is_test else 'Valid',
                    'Iter: [%d/%d]' % (batch_idx + 1, len(loader)),
                    'Time %.3f (%.3f)' % (batch_time.val, batch_time.avg),
                    'Loss %.4f (%.4f)' % (losses.val, losses.avg),
                    'Error %.4f (%.4f)' % (error.val*100, error.avg*100),
                    'Acc %.4f (%.4f)' % (acc.val*100, acc.avg*100),
                ])
                print(res)

    # Return summary statistics
    return batch_time.avg, losses.avg, error.avg,acc.avg

def train(model, args):
    if args.seed is not None:
        torch.manual_seed(args.seed)
    savePath = os.path.join(args.train_save,args.model)
    os.makedirs(savePath,exist_ok=True)
    # Data loaders
    train_loader, test_loader = get_readData(args)

    # Model on cuda
    if torch.cuda.is_available():
        model = model.cuda()

    # Wrap model for multi-GPUs, if necessary
    model_wrapper = model
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model_wrapper = torch.nn.DataParallel(model).cuda()

    # Optimizer
    # optimizer = torch.optim.Adam(model_wrapper.parameters(), lr=lr, weight_decay=wd)
    optimizer = torch.optim.SGD(model_wrapper.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=args.weight_decay)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs],gamma=0.1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-2)


    # Start log
    with open(os.path.join(savePath, 'results.csv'), 'w') as f:
        f.write('epoch,train_loss,train_error,valid_loss,valid_error\n')

    # Train model
    best_acc = 0.
    TrainAcc = []
    TestAcc = []

    for epoch in range(args.epochs):
        _, train_loss, train_error = train_epoch(
            model=model_wrapper,
            loader=train_loader,
            optimizer=optimizer,
            epoch=epoch,
            n_epochs=args.epochs,
        )
        scheduler.step()
        _, valid_loss, valid_error,valid_acc = eval_epoch(
            model=model_wrapper,
            loader= test_loader,
            is_test=(True)
        )
        TrainAcc.append((1-train_error))
        TestAcc.append(valid_acc)
        # Determine if model is the best
        if valid_acc > best_acc:
            best_acc = valid_acc
            print('New best error: %.4f' % (best_acc*100))
            torch.save(model.state_dict(), os.path.join(savePath, args.model+'_model.pth'))


        # Log results
        with open(os.path.join(savePath, 'results.csv'), 'a') as f:
            f.write('%03d,%0.6f,%0.4f,%0.6f,%0.4f,\n' % (
                (epoch + 1),
                train_loss,
                train_error*100,
                valid_loss,
                valid_error*100,
            ))

    # Final test of model on test set
    model.load_state_dict(torch.load(os.path.join(savePath, args.model+'_model.pth')))
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model).cuda()

    test_results = eval_epoch(
        model=model,
        loader=test_loader,
        is_test=True
    )

    _, _, test_error,test_acc = test_results

    with open(os.path.join(savePath, 'results.csv'), 'a') as f:
        f.write('Test_Acc:%0.4f\n' % (test_acc*100))

    if args.drawing_picture == True:
        epoch = [i for i in range(1, args.epochs+1)]
        excelplot_acc(epoch,TrainAcc,TestAcc,savePath)


    print('Test_Final Acc :%0.4f' % (test_acc*100))