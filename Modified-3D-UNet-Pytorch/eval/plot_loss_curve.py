from evaluate_tools import plot_loss

# train_csv = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/bce/train_loss.csv'
# val_csv = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/bce/val_loss.csv'
# plot_loss(val_csv, train_csv)

# train_csv = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/bce_1/train_loss.csv'
# val_csv = '/data/ccusr/xinyug/annotation/Modified-3D-UNet-Pytorch/results/bce_1/val_loss.csv'
# plot_loss(val_csv, train_csv)

train_csv = './results/unet_attn_generalised_softdice/train_loss.csv'
val_csv = './results/unet_attn_generalised_softdice/val_loss.csv'
plot_loss(val_csv, train_csv)
