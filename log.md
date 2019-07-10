#### 2019年07月08日19:10:23

1. 将EyeNet作为父类，衍生出了两个子类EyeNet_ldmks 和EyeNet_gaze
2. 实现了EyeNet.load()函数和EyeNet.predict()函数
3. 实现了EyeNet_ldmks.predict()函数，输入一张RGB图片输出18个眼部关键点
4. 初始化了git仓库并push到了github上面。
5. 完善了eyenet_gaze的代码, 但是训练的效果不好, loss 一直在抖动完全没有下降的迹象。

#### 2019年07月09日12:04:35

1. train_gaze()老是不收敛，但是之前的old train是收敛的，所以现在打算将old_train函数改好，测试一下现在是否还是能收敛。
2. 没有修改old_train.py , 发现每一个iter之间的抖动是由于batch太小导致的, 将batch改成2之后, 在dev数据集上面10个epoch之后明显是可以下降的。

#### 2019年07月10日19:03:31
1. 昨晚将EyeNet_gaze通过transfer_learning 的方式在dev dataset上面跑了500个epoch, 在100个epoch的时候就已经完全过拟合了, 但是有一个问题,不知为何v_yaw_diff & v_pitch_diff 都有明显在下降, 但是loss反倒一直在上升。
2. 上述问题好像是因为在计算val_loss 的时候：loss = criterion(gaze_targets, gaze_pred) gaze_pred 没有加上.squeeze(). 测试中...  