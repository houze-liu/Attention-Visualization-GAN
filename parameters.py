import argparse


def str2bool(v):
    return v.lower() in ('true')


def get_parameters():

    parser = argparse.ArgumentParser()

    # Model hyper-parameters
    parser.add_argument('--imsize', type=int, default=64)
    parser.add_argument('--z_dim', type=int, default=128)
    parser.add_argument('--g_dim', type=int, default=64)
    parser.add_argument('--d_dim', type=int, default=64)
    parser.add_argument('--lambda_gp', type=float, default=10)
    parser.add_argument('--version', type=str, default='sagan1')

    # Training setting
    parser.add_argument('--total_step', type=int, default=200000, help='how many times to update the generator')
    parser.add_argument('--iter_start_decay', type=int, default=100000, help='iters starting to decay lr')
    parser.add_argument('--lr_iter_decay', type=str, default=1000, help='how many steps before every time updating lr')
    parser.add_argument('--resume_iter', type=int, default=None, help='resume training from this step')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--g_lr', type=float, default=0.0001)
    parser.add_argument('--d_lr', type=float, default=0.0004)
    parser.add_argument('--lr_decay', type=float, default=0.95)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Misc
    parser.add_argument('--train', type=str2bool, default=True)
    parser.add_argument('--dataset', type=str, default='celeb', choices=['celeb'])
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)

    # Path
    parser.add_argument('--image_path', type=str, default='./data')
    parser.add_argument('--model_save_path', type=str, default='./models')
    parser.add_argument('--sample_path', type=str, default='./samples')

    # Step size
    parser.add_argument('--log_step', type=int, default=20, help='print out log info per this steps')
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=float, default=4000)

    return parser.parse_args()
