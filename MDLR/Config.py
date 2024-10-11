import argparse

def load_config():
    parser = argparse.ArgumentParser()

    #Model Config
    parser.add_argument('--model', type=str,choices=['Botnet18','Botnet34','ResNet18','ResNet34','VGG16','densenet121','DBB_ResNet18','DBB_ResNet50'],default='ResNet18',help='Select model')

    # Config of FilePath
    parser.add_argument('--root-dir', type=str, default='..\Center_A', help='Root directory of the data')
    parser.add_argument('--txt-dir',  type=str, default='..\data_txt\Center_A', help='TXT Path for saving the text')
    parser.add_argument('--train-save', type=str, default='..\Train_save', help='Save the address during training')

    # Config of model Training
    parser.add_argument('--numclass', type=int, default=2, help='How many categories to choose')
    parser.add_argument('--category', type=list, default=['0', '1'], help='Two types of data labels')
    parser.add_argument('--PreTrain', type=bool, default=True, help='Choose to use a natural image pretraining model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=bool, default=False)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--CUDA', type=bool, default=True)
    parser.add_argument('--feature-Number', type=int, default=3904)
    parser.add_argument('--drawing-picture', type=bool, default=True,help='Draw the training curve')


    return parser.parse_args()
