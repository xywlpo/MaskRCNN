import os
import numpy as np
import cv2
import skimage.draw
import skimage.io
import json
from PIL import Image,ImageEnhance, ImageFilter

def json2mask(dataset_dir, flag):

    # get x and y coordinate by .json file
    annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
    annotations = list(annotations.values())

    # skip unannotated images
    annotations = [a for a in annotations if a['regions']]

    if flag == 0:
        train_path = "./Output/train"
        if os.path.exists(train_path) == False:
            os.makedirs(train_path)
    else:
        val_path = "./Output/val"
        if os.path.exists(val_path) == False:
            os.makedirs(val_path)

    idx = 0
    for a in annotations:
        idx = idx + 1
        polygons = [r['shape_attributes'] for r in a['regions'].values()]
        polygons_labels = [r['region_attributes'] for r in a['regions'].values()]
        image_path = os.path.join(dataset_dir, a['filename'])
        image = skimage.io.imread(image_path)
        height, width = image.shape[:2]
        cv2.imshow('x', image[:,:,2])
        cv2.waitKey(0)
        break
        mask = np.zeros([height, width, len(polygons)], dtype=np.uint8)

        if flag == 0:
            sub_path_org = train_path + "/" + str(idx) + "/images"
            sub_path_mask = train_path + "/" + str(idx) + "/masks"
        else:
            sub_path_org = val_path + "/" + str(idx) + "/images"
            sub_path_mask = val_path + "/" + str(idx) + "/masks"

        if os.path.exists(sub_path_org) == False:
            os.makedirs(sub_path_org)
        if os.path.exists(sub_path_mask) == False:
            os.makedirs(sub_path_mask)

        print("正在处理第：", idx, "张图片！")

        # 存储原图
        dst = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sub_path_org + "/" + str(idx) + ".png", dst)

        for i, p in enumerate(polygons):
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 255
            # cv2.imshow("test", mask[:,:,i])
            # cv2.waitKey(200)
            filename = sub_path_mask + "/" + polygons_labels[i]['class'] + "_" + str(i) + ".png"
            cv2.imwrite(filename, mask[:,:,i])

def randomFlip(img, randIdx):
    # randIdx = np.random.randint(0, 1)
    if randIdx == 0:
        out = img.transpose(Image.FLIP_LEFT_RIGHT)
    else:
        out = img.transpose(Image.FLIP_TOP_BOTTOM)
    return out

def randomRotation(img, rand_angle):
    # rand_angle = np.random.randint(1, 360)
    return img.rotate(rand_angle, Image.BICUBIC)

def randomColor(img):
    random_factor = np.random.randint(7, 15) / 10.  # 随机因子
    color_image = ImageEnhance.Color(img).enhance(random_factor)  # 调整图像的饱和度
    random_factor = np.random.randint(8, 13) / 10.  # 随机因子
    brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
    random_factor = np.random.randint(8, 12) / 10.  # 随机因1子
    contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
    random_factor = np.random.randint(8, 18) / 10.  # 随机因子
    sharp_image = ImageEnhance.Sharpness(contrast_image).enhance(random_factor)     # 调整图像锐化度
    return sharp_image# 调整图像锐度

def ImageAugment(dataset_dir):
    ###########################################################################
    ############################## 图像随机翻转 ################# OK ##########
    ###########################################################################
    file_dir = os.walk(dataset_dir)
    for f in next(file_dir)[1]:
        randIdx = np.random.randint(0, 2)  # 以1/2的概率生成翻转
        #randIdx = 1
        if randIdx == 1:
            print("图像随机翻转处理：", f)
            org_image_path = dataset_dir + '/' + f + '/images/' + f + '.png'
            mask_image_path = dataset_dir + '/' + f + '/masks/'
            org_image_path_aug = dataset_dir + '/' + f + '_flip' + '/images/'
            mask_image_path_aug = dataset_dir + '/' + f + '_flip' + '/masks/'

            if os.path.exists(org_image_path_aug) == False:
                os.makedirs(org_image_path_aug)
            if os.path.exists(mask_image_path_aug) == False:
                os.makedirs(mask_image_path_aug)
                
            randIdx = np.random.randint(0, 1)   # Image Filp
            org_img = Image.open(org_image_path, "r")
            org_image_flip = randomFlip(org_img, randIdx)
            org_image_flip.save(org_image_path_aug + f + '_flip' + '.png')
            for mask_file in next(os.walk(mask_image_path))[2]:
                mask_name = mask_file.split('.')
                if mask_file.endswith(".png"):
                    mask_image = Image.open(mask_image_path + mask_file, "r")
                    mask_image_flip = randomFlip(mask_image, randIdx)
                    mask_dst = mask_image_path_aug + mask_name[0] + '_flip' + '.png'
                    mask_image_flip.save(mask_dst)
    print("图像随机翻转处理完毕！")

    ###########################################################################
    ############################## 图像旋转 ################# OK ##############
    ###########################################################################
    file_dir = os.walk(dataset_dir)
    for f in next(file_dir)[1]:
        randIdx = np.random.randint(0, 3)  # 以1/3的概率计算旋转
        #randIdx = 1
        if randIdx == 1:
            print("图像随机旋转处理：", f)
            org_image_path = dataset_dir + '/' + f + '/images/' + f + '.png'
            mask_image_path = dataset_dir + '/' + f + '/masks/'

            # 计算旋转
            rotation_num = 40
            rand_angle = 0
            for counts in range(rotation_num):
                #randSel = np.random.randint(0, 3)
                randSel = 2
                if randSel == 2:
                    rand_angle = (counts + 1) * 360.0/rotation_num
                    org_image_path_aug = dataset_dir + '/' + f + '_rot' + str(counts) + '/images/'
                    mask_image_path_aug = dataset_dir + '/' + f + '_rot' + str(counts) + '/masks/'
                    if os.path.exists(org_image_path_aug) == False:
                        os.makedirs(org_image_path_aug)
                    if os.path.exists(mask_image_path_aug) == False:
                        os.makedirs(mask_image_path_aug)

                    org_img = Image.open(org_image_path, "r")
                    org_img_rot = randomRotation(org_img, rand_angle)
                    org_img_rot.save(org_image_path_aug + f + '_rot' + str(counts) + '.png')

                    for mask_file in next(os.walk(mask_image_path))[2]:
                        mask_name = mask_file.split('.')
                        if mask_file.endswith(".png"):
                            mask_image = Image.open(mask_image_path + mask_file, "r")
                            mask_img_rot = randomRotation(mask_image, rand_angle)
                            mask_dst = mask_image_path_aug + mask_name[0] + '_rot' + str(counts) + '.png'
                            mask_img_rot.save(mask_dst)
    print("图像随机旋转处理完毕！")
    
    ###########################################################################
    ############################## 图像随机缩放 ##############################
    ###########################################################################
    file_dir = os.walk(dataset_dir)
    width = 2064
    height = 1544
    for f in next(file_dir)[1]:
        randIdx = np.random.randint(0, 2)  # 以1/4的概率计算缩放
        #randIdx = 1
        if randIdx == 1:
            print("图像随机缩放：", f)
            org_image_path = dataset_dir + '/' + f + '/images/' + f + '.png'
            mask_image_path = dataset_dir + '/' + f + '/masks/'

            randChoose = np.random.randint(0, 2)
            if randChoose == 0:
                ratio = 1.08
                num = 2                

                # 放大图像
                for counts in range(num):
                    re_width = width * ratio;
                    re_height = height * ratio;

                    # 创建文件保存路径
                    org_image_path_aug = dataset_dir + '/' + f + '_enlarge' + str(counts) + '/images/'
                    mask_image_path_aug = dataset_dir + '/' + f + '_enlarge' + str(counts) + '/masks/'
                    if os.path.exists(org_image_path_aug) == False:
                        os.makedirs(org_image_path_aug)
                    if os.path.exists(mask_image_path_aug) == False:
                        os.makedirs(mask_image_path_aug)

                    # 原始图像处理
                    org_img = Image.open(org_image_path, "r")
                    org_img = org_img.resize((int(re_width), int(re_height)), Image.ANTIALIAS)
                    org_crop = org_img.crop((int(re_width/2-width/2), int(re_height/2-height/2), int(re_width/2+width/2), int(re_height/2+height/2)))
                    org_crop.save(org_image_path_aug + f + '_enlarge' + str(counts) + '.png')

                    for mask_file in next(os.walk(mask_image_path))[2]:
                        mask_name = mask_file.split('.')
                        if mask_file.endswith(".png"):
                            mask_image = Image.open(mask_image_path + mask_file, "r")
                            mask_image = mask_image.resize((int(re_width), int(re_height)), Image.ANTIALIAS)
                            mask_crop = mask_image.crop((int(re_width/2-width/2), int(re_height/2-height/2), int(re_width/2+width/2), int(re_height/2+height/2)))
                            mask_dst = mask_image_path_aug + mask_name[0] + '_enlarge' + str(counts) + '.png'
                            mask_crop.save(mask_dst)

            if randChoose == 1:
                # 缩小图像
                ratio = 1.08
                num = 2
                for counts in range(num):
                    re_width = int(width / ratio);
                    re_height = int(height / ratio);

                    # 创建文件保存路径
                    org_image_path_aug = dataset_dir + '/' + f + '_smaller' + str(counts) + '/images/'
                    mask_image_path_aug = dataset_dir + '/' + f + '_smaller' + str(counts) + '/masks/'
                    if os.path.exists(org_image_path_aug) == False:
                        os.makedirs(org_image_path_aug)
                    if os.path.exists(mask_image_path_aug) == False:
                        os.makedirs(mask_image_path_aug)

                    # 原始图像处理
                    org_img = Image.open(org_image_path, "r")
                    org_img = org_img.resize((re_width, re_height), Image.ANTIALIAS)
                    black_ground = Image.new('RGB', (width, height), (0, 0, 0))
                    black_ground.paste(org_img, (0, 0))
                    black_ground.save(org_image_path_aug + f + '_smaller' + str(counts) + '.png')

                    for mask_file in next(os.walk(mask_image_path))[2]:
                        mask_name = mask_file.split('.')
                        if mask_file.endswith(".png"):
                            mask_image = Image.open(mask_image_path + mask_file, "r")
                            mask_image = mask_image.resize((re_width, re_height), Image.ANTIALIAS)
                            mask_ground = Image.new('RGB', (width, height), (0, 0, 0))
                            mask_ground.paste(mask_image, (0, 0))
                            mask_dst = mask_image_path_aug + mask_name[0] + '_smaller' + str(counts) + '.png'
                            mask_ground.save(mask_dst)
    print("图像缩放处理完毕！")

    ###########################################################################
    ############################## 图像模糊处理 ########## OK #################
    ###########################################################################
    file_dir = os.walk(dataset_dir)
    for f in next(file_dir)[1]:
        randIdx = np.random.randint(0, 8)  # 以1/10的概率计算高斯模糊
        if randIdx == 3:
            print("图像模糊处理：", f)
            org_image_path = dataset_dir + '/' + f + '/images/' + f + '.png'
            org_img = Image.open(org_image_path, "r")
            org_img_dou = org_img.filter(ImageFilter.GaussianBlur(radius=1))
            org_img_dou.save(org_image_path)
    print("图像模糊处理完毕")

    ###########################################################################
    ############################## 图像颜色抖动 ############## OK ############
    ###########################################################################
    file_dir = os.walk(dataset_dir)
    for f in next(file_dir)[1]:
        randIdx = np.random.randint(0, 4)  # 以1/5的概率计算颜色抖动
        if randIdx == 3:
            print("图像颜色抖动处理：", f)
            org_image_path = dataset_dir + '/' + f + '/images/' + f + '.png'
            org_img = Image.open(org_image_path, "r")
            org_img_dou = randomColor(org_img)
            org_img_dou.save(org_image_path)
    print("图像颜色抖动处理完毕")

if __name__ == '__main__':

    ## convert json file to mask image
    train_dataset = "./yuanpan/train"
    val_dataset = "./yuanpan/val"
    json2mask(train_dataset, 0)
    json2mask(val_dataset, 1)

    ## 图像增广
    ImageAugment("./Output/train")
    ImageAugment("./Output/val")
    print("图像增广处理完毕！")





