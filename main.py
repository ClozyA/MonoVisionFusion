import cv2
import numpy as np
import os

# 1. 读取图片
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(img)
    return images

# 2. 图像预处理
def preprocess_image(image):
    # 转为灰度图
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 直方图均衡化
    equalized_image = cv2.equalizeHist(gray_image)
    return equalized_image

# 3. 图像配准
def register_images(img1, img2):
    # 检测特征点和描述符 (ORB)
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # 使用BFMatcher进行特征匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)

    # 提取匹配点
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    return M

# 4. 图像融合
def fuse_images(img1, img2, M):
    # 获取img2的尺寸
    h, w, _ = img2.shape
    # 对img1进行透视变换
    warped_img1 = cv2.warpPerspective(img1, M, (w, h))
    # 图像融合 (加权平均)
    fused_image = cv2.addWeighted(warped_img1, 0.5, img2, 0.5, 0)
    return fused_image


# 4. 图像融合 (去掉遮罩)
def fuse_images_without_mask(img1, img2, M):
    # 获取img2的尺寸
    h, w, _ = img2.shape
    # 对img1进行透视变换
    warped_img1 = cv2.warpPerspective(img1, M, (w, h))

    # 创建无遮罩的融合图像
    fused_image = np.where(warped_img1 > 0, warped_img1, img2)
    return fused_image


# 主流程
def main():
    folder_path = "./pic"
    images = load_images_from_folder(folder_path)

    if len(images) < 2:
        print("文件夹内图片数量不足两张")
        return

    # 读取两张图片
    img1, img2 = images[0], images[1]

    # 图像预处理
    preprocessed_img1 = preprocess_image(img1)
    preprocessed_img2 = preprocess_image(img2)

    # 图像配准
    M = register_images(preprocessed_img1, preprocessed_img2)

    # 图像融合
    fused_image = fuse_images(img1, img2, M)
    fused_image_without_mask = fuse_images_without_mask(img1, img2, M)

    # 显示结果
    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    cv2.imshow("Fused Image with Mask", fused_image)
    cv2.imshow("Fused Image without Mask", fused_image_without_mask)

    # 保存结果
    cv2.imwrite("./fused_image.jpg", fused_image)
    cv2.imwrite("./fused_image_without_mask.jpg", fused_image_without_mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
