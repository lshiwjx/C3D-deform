import os
import random
import skimage.data as ski
import numpy as np

data_path = '/home/sl/Resource/UCF101'
project_path = './'
random_num = 4

# make train.txt and val.txt
data_dirs = sorted(os.listdir(data_path))
train_list = []
val_list = []
label = -1
count_for_train = 0
count_for_val = 0
mean_list = []
label_list = []

for data_dir in data_dirs:
    video_path = os.path.join(data_path, data_dir)
    video_dirs = sorted(os.listdir(video_path))
    label += 1
    label_list.append(data_dir)
    for video_dir in video_dirs:
        print("now for data dir %s\n" % video_dir)
        img_path = os.path.join(video_path, video_dir)
        if not random.randint(0, random_num) == random_num:
            train_list.append((img_path, label))
            count_for_train += 1
        else:
            val_list.append((img_path, label))
            count_for_val += 1
        # calculate mean of img
        imgs = sorted(os.listdir(img_path))
        for img in imgs:
            pic_path = os.path.join(img_path, img)
            pic = ski.imread(pic_path)
            mean_list.append(np.mean(pic, (0, 1)))

os.chdir(project_path)
train_file = open('train.txt', 'w')
for item in train_list:
    train_file.write("%s %s \n" % (item[0], item[1]))
train_file.close()

val_file = open('val.txt', 'w')
for item in val_list:
    val_file.write("%s %s \n" % (item[0], item[1]))
val_file.close()

label_file = open('label.txt', 'w')
for item in label_list:
    label_file.write("%s \n" % item)
label_file.close()

mean = np.mean(np.array(mean_list), 0)
others_file = open('others.txt','w')
others_file.write("mean : %.2f %.2f %.2f\n" % (mean[0],mean[1],mean[2]))
others_file.write("train num: %d\n" % count_for_train)
others_file.write("val num: %d\n" % count_for_val)
print("train num: ", count_for_train)
print("val num: ", count_for_val)
print("mean val: ", mean)
