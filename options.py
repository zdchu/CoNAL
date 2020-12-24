def model_opts(parser):
    parser.add_argument('--dropout', default=0.5, type=float)
    parser.add_argument('--lr', default=0.01, type=float)
    parser.add_argument('--num_epochs', default=100, type=int)
    parser.add_argument('--mode', default='train')

    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--num_samples', type=int, default=40000)
    parser.add_argument('--noise_rate', type=float, default=0.4)
    parser.add_argument('--num_users', type=int, default=30)