# ==============================================================================
# This file is part of a modified distribution of TimeMixer.
# Modifications copyright (c) 2026 Anastasios Papadopoulos, Apostolos Giannoulidis.
# Original code licensed under Apache License, Version 2.0.
# ==============================================================================

from method.timemixerpp.TimeMixer.data_provider.data_loader import Dataset_Custom
from method.timemixerpp.TimeMixer.data_provider.uea import collate_fn
from torch.utils.data import DataLoader

data_dict = {
    'custom': Dataset_Custom,
}


def data_provider(args, flag, df, vali_points_percentage):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        pass
    #     drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         win_size=args.seq_len,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last)
    #     return data_set, data_loader
    # elif args.task_name == 'classification':
    #     drop_last = False
    #     data_set = Data(
    #         root_path=args.root_path,
    #         flag=flag,
    #     )
    #     print(flag, len(data_set))
    #     data_loader = DataLoader(
    #         data_set,
    #         batch_size=batch_size,
    #         shuffle=shuffle_flag,
    #         num_workers=args.num_workers,
    #         drop_last=drop_last,
    #         collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
    #     )
    #     return data_set, data_loader
    else:
        # if args.data == 'm4':
        #     drop_last = False
        data_set = Data(

            df,
            # root_path=args.root_path,
            # data_path=args.data_path,
            vali_points_percentage=vali_points_percentage,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
