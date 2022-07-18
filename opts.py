import argparse
# '''
parser = argparse.ArgumentParser(description='PyTorch ISIC2019 CNN Training')
parser.add_argument('--use_cuda', type=bool, default=True, help='device to train on')
parser.add_argument('--distribution', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs to train on') 
parser.add_argument('--param', default='test', type=str, help='test')
parser.add_argument('--ratio_finetune', default=75, type=float, help='ratio of last layer to finetune') 
parser.add_argument('--weighting_type', default='CS', type=str, help='data sampling strategy for train loader')
parser.add_argument('--task', default='multi', type=str, help='task case')
parser.add_argument('--loss_type', default="CE", type=str, help='loss type')
parser.add_argument('--max_lr', '--learning-rate', default=0.001, type=float, help='define the maximum learning rate during cyclical learning. For more details check the backup implementation in  this file',dest='max_lr')
parser.add_argument('-b','--batch_size', default=64, type=int, help='define batch size') 
parser.add_argument('--delay', default=15, type=int,help='number of epoch to patience before early stopping the training if no improvement of balanced accuracy') 
parser.add_argument('--num_runs', default=10, type=int, help='number of runs to launch ')
parser.add_argument('--model_path', default='/checkpoint/', type=str, help='checkpoint path ')
parser.add_argument('--history_path', default='/history/', type=str, help='history path ')
parser.add_argument('--result_dir', default='/results/', type=str, help='results path ')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=2e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')




