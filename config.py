# # path to your own data and coco file
train_data_dir = "./data/frc/train"
train_coco = "./data/frc/train/_annotations.coco.json"

# train_data_dir = "data/train"
# train_coco = "data/my_train_coco.json"

# Batch size
train_batch_size = 1

# Params for dataloader
train_shuffle_dl = False
num_workers_dl = 4

# Params for training

# Two classes; Only target class or background
num_classes = 9
num_epochs = 5
num_images = 50

lr = 0.005
momentum = 0.7
weight_decay = 0.005

# for inference
save_model_name = 'result/last_model.pth'
result_img_dir = 'result/imgs'
detection_threshold = 0.8
test_data_dir = 'data/test'
test_img_format = 'png'

