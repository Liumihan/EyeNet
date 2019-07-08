class Config(object):

    epochs = 2000
    lr = 5e-5
    weight_decay = 1e-5
    downsample_scale = 1
    gau_size = 7
    dev = 100
    # UnityEyes dataset
    train_data_dir = "/media/liumihan/HDD_Documents/眼部数据集/100wUnityEyes/train"
    val_data_dir = "/media/liumihan/HDD_Documents/眼部数据集/100wUnityEyes/val"


    # MPIIGaze dataset
    MPIIGaze_img_dir = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/imgs'
    MPIIGaze_train_txt = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/left_one_out/p02_leftout/with_out_p02.txt'
    MPIIGaze_val_txt = '/media/liumihan/HDD_Documents/眼部数据集/MPIIGaze/Data/Normalized/left_one_out/p02_leftout/only_p02.txt'


    device = 'cuda:0'

    weight_save_dir = './weights/'
    # 如果想从头训练的话就将他置为None
    # checkpoint_path = 'weights/ELG_epoch38.pth'
    checkpoint_path = None
    saving_prefix = 'EyeNet_ldmk-0.1'

    batch_size = 2
    plot_every_iter = 3


opt = Config()
