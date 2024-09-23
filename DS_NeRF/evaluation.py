import pyiqa
import torch
import os
import numpy as np
from PIL import Image
import cv2

img2mse = lambda x, y: np.mean((x - y) ** 2)
img2l1 = lambda x, y: np.mean(np.abs(x - y))

def load_images_from_folder(folder):
    images = []
    filenames = []
    for filename in os.listdir(folder):
        # img = Image.open(os.path.join(folder, filename))
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames

def load_depths_from_folder(folder):
    depths = []
    filenames = []
    for filename in os.listdir(folder):
        depth = np.fromfile(os.path.join(folder, filename), dtype='float32')
        print('--------------depth: ', depth)
        print('--------------depth.shape: ', depth.shape)
        # depth = depth.reshape(192, 256) / 255.   
        if depth is not None:
            depths.append(depth)
            filenames.append(filename)
    return depths, filenames

def load_masks_from_folder(mask_folder):
    images = []
    filenames = []
    for filename in os.listdir(mask_folder):
        # img = Image.open(os.path.join(mask_folder, filename))
        img = cv2.imread(os.path.join(mask_folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images, filenames


# 0. list all available metrics
print(pyiqa.list_models())
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# 1. define required metrics
iqa_metric_psnr = pyiqa.create_metric('psnr', device=device)
iqa_metric_lpips = pyiqa.create_metric('lpips', device=device)
iqa_metric_fid = pyiqa.create_metric('fid', device=device)

# 2.Evaluation
sum_sum_score_psnr = 0.0
sum_sum_score_lpips = 0.0
sum_sum_score_fid = 0.0
sum_sum_score_l1 = 0.0
sum_sum_score_l2 = 0.0
for folder_number in range(1, 17):
    print('--------------------processing scene_', folder_number)
    sum_score_psnr = 0.0
    sum_score_lpips = 0.0
    sum_score_fid = 0.0
    sum_score_l1 = 0.0
    sum_score_l2 = 0.0

    # Load images from two folders
    folder_name = f'{folder_number:03}'  # Convert to three-digit folder name
    predicted_folder_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/rgb'
    predicted_images, predicted_filenames  = load_images_from_folder(predicted_folder_path)

    GT_folder_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/images'
    GT_images, GT_filenames  = load_images_from_folder(GT_folder_path)

    predicted_depth_folder_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/depth_img'
    predicted_dept_images, predicted_dept_filenames  = load_images_from_folder(predicted_depth_folder_path)

    GT_depth_folder_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/depth_GT'
    GT_dept_images, GT_dept_filenames  = load_images_from_folder(GT_depth_folder_path)

    # psnr & lpips
    for i in range(len(predicted_images)):
        predicted_image_name = os.path.join(predicted_folder_path, predicted_filenames[i])
        GT_image_name = os.path.join(GT_folder_path, GT_filenames[i])

        score1 = iqa_metric_psnr(predicted_image_name, GT_image_name)
        sum_score_psnr += score1

        score2 = iqa_metric_lpips(predicted_image_name, GT_image_name)
        sum_score_lpips += score2
    print('---------score_psnr: ', sum_score_psnr/len(predicted_images))
    print('---------score_lpips: ', sum_score_lpips/len(predicted_images))

    # FID
    predicted_image_name = os.path.join(predicted_folder_path)
    GT_image_name = os.path.join(GT_folder_path)
    sum_score_fid = iqa_metric_fid(predicted_image_name, GT_image_name)
    print('---------score_fid: ', sum_score_fid)

    # Depth L1 & L2
    for i in range(len(predicted_images)):
        predicted_dept_image_i = predicted_dept_images[i]
        GT_dept_image_i = GT_dept_images[i]
        sum_score_l2 += img2mse(predicted_dept_image_i, GT_dept_image_i)
        sum_score_l1 += img2l1(predicted_dept_image_i, GT_dept_image_i)
    print('---------sum_score_l2: ', sum_score_l2/len(predicted_images))
    print('---------sum_score_l1: ', sum_score_l1/len(predicted_images))

    # save as txt
    sum_sum_score_psnr += sum_score_psnr/len(predicted_images)
    sum_sum_score_lpips += sum_score_lpips/len(predicted_images)
    sum_sum_score_fid += sum_score_fid
    sum_sum_score_l2 += sum_score_l2/len(predicted_images)
    sum_sum_score_l1 += sum_score_l1/len(predicted_images)
    save_file_path = f'/mnt/lustre/hhchen/SPIn-NeRF-master/logs/{folder_name}/testset_010000/eval.txt'
    with open(save_file_path, 'w') as file:
        file.write(str(sum_score_psnr/len(predicted_images)) + '\n')  # Write each value on a new line
        file.write(str(sum_score_lpips/len(predicted_images)) + '\n')  # Write each value on a new line
        file.write(str(sum_score_fid) + '\n')  # Write each value on a new line
        file.write(str(sum_score_l2) + '\n')  # Write each value on a new line
        file.write(str(sum_score_l1) + '\n')  # Write each value on a new line



print('---------sum_sum_score_psnr: ', sum_sum_score_psnr/16)
print('---------sum_sum_score_lpips: ', sum_sum_score_lpips/16)
print('---------sum_sum_score_fid: ', sum_sum_score_fid/16)
print('---------sum_sum_score_l2: ', sum_sum_score_l2/16)
print('---------sum_sum_score_l1: ', sum_sum_score_l1/16)



















 # mask_images, mask_filenames = load_masks_from_folder('/mnt/lustre/hhchen/SPIn-NeRF-master/logs/001/testset_010000/masks')
    # Check if number of original images and masks are the same
    # assert len(original_images) == len(mask_images), "Number of original images and masks should be the same."

    # 2. save masked images
    # for i in range(len(predicted_images)):
    #     predicted_image_i = predicted_images[i]
    #     GT_image_i = GT_images[i]
    #     mask_image_i = mask_images[i]

    #     # 3. obtain the masked bounding box
    #     contours, _ = cv2.findContours(mask_image_i, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # Iterate through the contours and calculate pixel counts within each bounding box
    #     for contour in contours:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         # Crop the bounding box region from the mask image
    #         predicted_image_i_masked = predicted_image_i[y:y + h, x:x + w]
    #         GT_image_i_masked = GT_image_i[y:y + h, x:x + w]

    #         predicted_image_i_masked_name = os.path.join('/mnt/lustre/hhchen/SPIn-NeRF-master/logs/001/testset_010000/RGB_Spin_results_masked/', mask_filenames[i])
    #         cv2.imwrite(predicted_image_i_masked_name, predicted_image_i_masked, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    #         GT_image_i_masked_masked_name = os.path.join('/mnt/lustre/hhchen/SPIn-NeRF-master/logs/001/testset_010000/RGB_GT_masked/', mask_filenames[i])
    #         cv2.imwrite(GT_image_i_masked_masked_name, GT_image_i_masked, [cv2.IMWRITE_PNG_COMPRESSION, 0])