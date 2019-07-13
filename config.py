class Config(object):

    epochs = 20
    lr = 2e-4
    weight_decay = 1e-4
    downsample_scale = 1
    gau_size = 7
    dev = 100
    # UnityEyes dataset
    train_data_dir = "/media/liumihan/HDD_Documents/眼部数据集/100wUnityEyes/train"
    val_data_dir = "/media/liumihan/HDD_Documents/眼部数据集/100wUnityEyes/val"

    # train_data_dir = '/home/luoyc/Daihuanhuan/datasets/eye/UnityEyes100w'
    # val_data_dir = '/home/luoyc/Daihuanhuan/datasets/eye/UnityEyes/val'

    # MPIIGaze dataset
    MPIIGaze_img_dir = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/imgs'
    # 全部
    MPIIGaze_train_txt = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/all.txt'
    # P00 left_out
    MPIIGaze_train_txt = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/left_one_out/p00_leftout/with_out_p01.txt'
    MPIIGaze_val_txt = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/left_one_out/p00_leftout/only_p01.txt'

    device = 'cuda:0'

    weight_save_dir = './weights/'
    # 如果想从头训练的话就将他置为None
    # checkpoint_path = 'weights/ELG_epoch38.pth'
    # checkpoint_path = "weights/EyeNet-0.1_epoch1.pth"
    checkpoint_path = None
    # saving_prefix = 'EyeNet_ldmk-0.1'
    saving_prefix = 'EyeNet_gaze-MPIIGaze_01'
    batch_size = 32  # batch_size 必须大于1, 不然在计算loss的时候Batch这个维度会被squeeze掉
    plot_every_iter = 500


opt = Config()
