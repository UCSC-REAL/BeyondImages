from utils.parser import parse_args
def run(args):
    from run_nn import run as run_nn
    choices=[
        'heart', 'banana',  'titanic', 'splice', 'twonorm', 'waveform', 'flare-solar'
    ]
    

    T_err_rec_ours_X = []
    T_err_rec_ours_Z = []
    for args.dataset in choices:
        error_ours_noisy, error_ours_clean = run_nn(args)

        T_err_rec_ours_X += [error_ours_noisy]
        T_err_rec_ours_Z += [error_ours_clean]

    print(f'HOC error weight X:\n{T_err_rec_ours_X}')
    print(f'HOC error weight Z:\n{T_err_rec_ours_Z}')

if __name__ == '__main__':
    run(parse_args())
