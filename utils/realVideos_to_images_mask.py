import cv2
import os
import math
from tqdm import tqdm
from multiprocessing import cpu_count, Pool
import pandas as pd  # 导入pandas库用于处理CSV文件


def load_video(video_path):
    """
    加载视频文件并返回视频捕获对象和总帧数
    :param video_path: 视频路径
    :return: 视频捕获对象和总帧数
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 获取总帧数
    return cap, total_frames


def calculate_frame_interval(total_frames, desired_intervals):
    """
    动态计算帧间隔
    :param total_frames: 视频的总帧数
    :param desired_intervals: 期望的检测次数
    :return: 动态间隔的帧数
    """
    return max(1, math.ceil(total_frames / desired_intervals))


def save_face(aligned_face, save_dir, frame_index):
    """
    保存对齐后的人脸
    :param aligned_face: 对齐后的人脸图像
    :param save_dir: 保存路径
    :param frame_index: 当前帧索引
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"frame_{frame_index:04d}.jpg")
    cv2.imwrite(save_path, aligned_face)


def expand_mask(frame_fake, x, y, w, h, target_size=224):
    """
    将边界框扩展到固定大小 224x224，并确保不会超出帧的尺寸。

    参数:
    - frame_fake: 输入帧 (numpy 数组)
    - x, y, w, h: 初始边界框的位置和大小
    - target_size: 目标扩展大小 (默认 224)

    返回:
    - expanded_x, expanded_y, expanded_w, expanded_h: 扩展后的边界框坐标和大小
    """
    # 确定目标框的中心位置
    center_x = x + w // 2
    center_y = y + h // 2

    # 计算扩展后的边界框的左上角坐标
    expanded_x = max(0, center_x - target_size // 2)
    expanded_y = max(0, center_y - target_size // 2)

    # 保证扩展区域不会超出帧的尺寸
    if expanded_x + target_size > frame_fake.shape[1]:
        expanded_x = frame_fake.shape[1] - target_size
    if expanded_y + target_size > frame_fake.shape[0]:
        expanded_y = frame_fake.shape[0] - target_size

    # 确保边界框尺寸为目标大小
    expanded_w = target_size
    expanded_h = target_size

    return expanded_x, expanded_y, expanded_w, expanded_h

def process_single_video(args):
    """
    处理单个视频文件
    :param args: 参数列表，包含真实视频路径、掩膜视频路径、保存路径、目标图像大小、期望采集帧数
    """
    real_video_path, mask_video_path, save_dir, target_size, desired_intervals = args

    # 加载视频并检查帧数是否一致
    cap_real, total_frames_real = load_video(real_video_path)
    cap_mask, total_frames_mask = load_video(mask_video_path)

    print(f"正在处理视频: {os.path.basename(real_video_path)}，总帧数: {total_frames_real}")

    # 动态计算帧间隔
    frame_interval = calculate_frame_interval(total_frames_real, desired_intervals)

    # 计算采集的总帧数
    frames_to_process = math.ceil(total_frames_real / frame_interval)

    frame_index = 0
    frames_collected = 0

    # 使用 tqdm 显示进度条
    with tqdm(total=frames_to_process, desc=f"Processing {os.path.basename(real_video_path)}", unit="frame") as pbar:
        while True:
            ret_real, frame_real = cap_real.read()
            ret_mask, frame_mask = cap_mask.read()

            if not ret_real or not ret_mask:
                break

            # 只处理指定间隔的帧
            if frame_index % frame_interval == 0:
                # 转为灰度图并创建二进制掩膜
                gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # 获取掩模区域的边界框
                x, y, w, h = cv2.boundingRect(binary_mask)
                expanded_x, expanded_y, expanded_w, expanded_h = expand_mask(frame_real, x, y, w, h, target_size=target_size)

                # 从伪造帧中提取扩展区域
                face_img = frame_real[expanded_y:expanded_y + expanded_h,
                                      expanded_x:expanded_x + expanded_w]

                # face_img = frame_real[y:y + h, x:x + w]

                # # 根据目标尺寸缩放提取的人脸图像
                # face_img = cv2.resize(face_img, (target_size, target_size))

                save_face(face_img, save_dir, frame_index)
                frames_collected += 1  # 每采集一帧，更新已采集帧数

                # 更新进度条
                pbar.update(1)

            frame_index += 1

        # 检查是否已采集的帧数小于 desired_intervals，若小于则采集最后一帧
        if frames_collected < desired_intervals:
            # 确保采集最后一帧
            cap_real.set(cv2.CAP_PROP_POS_FRAMES, total_frames_real - 1)  # 跳到最后一帧
            cap_mask.set(cv2.CAP_PROP_POS_FRAMES, total_frames_real - 1)

            ret_real, frame_real = cap_real.read()
            ret_mask, frame_mask = cap_mask.read()

            if ret_real and ret_mask:
                # 转为灰度图并创建二进制掩膜
                gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # 获取掩模区域的边界框
                x, y, w, h = cv2.boundingRect(binary_mask)
                expanded_x, expanded_y, expanded_w, expanded_h = expand_mask(frame_real, x, y, w, h, target_size=target_size)

                # 从伪造帧中提取扩展区域
                face_img = frame_real[expanded_y:expanded_y + expanded_h,
                                      expanded_x:expanded_x + expanded_w]
                
                # face_img = frame_real[y:y + h, x:x + w]

                # # 根据目标尺寸缩放提取的人脸图像
                # face_img = cv2.resize(face_img, (target_size, target_size))

                save_face(face_img, save_dir, total_frames_real - 1)  # 保存最后一帧
                frames_collected += 1  # 更新已采集的帧数

                pbar.update(1)

    cap_real.release()
    cap_mask.release()



def process_videos(real_dir, mask_dir, save_dir_base, mode,target_size,desired_intervals):
    """
    处理目录下的所有视频文件
    :param real_dir: 真实视频路径
    :param mask_dir: 掩膜视频路径
    :param save_dir_base: 保存路径基目录
    :param mode: 处理模式，1 为单进程，2 为多进程
    """

    # 获取真实视频和目标视频的文件名列表
    real_files = sorted(os.listdir(real_dir))  # 获取真实视频文件名列表
    mask_files = sorted(os.listdir(mask_dir))  # 获取目标视频文件名列表

    tasks = []  # 初始化任务列表

    # 遍历真实视频文件名列表
    for real_file in real_files:
        # 提取 mask_files 中的原始视频名部分，并检查是否与 real_file 匹配
        for mask_file in mask_files:
            # 分离 mask_file 的原始视频名部分
            original_name = mask_file.split("_")[0]  # 提取 '原始视频名'

            # 如果原始视频名与 real_file 匹配
            if original_name == os.path.splitext(real_file)[0]:  # 去除扩展名进行比较
                # 构造文件路径
                real_path = os.path.join(real_dir, real_file)  # 真实视频路径
                mask_path = os.path.join(mask_dir, mask_file)  # 目标视频路径
                save_dir = os.path.join(save_dir_base, os.path.splitext(real_file)[0])  # 保存路径

                # 添加任务到任务列表
                tasks.append((real_path, mask_path, save_dir,target_size,desired_intervals))
                break

    if mode == 1:
        print("单进程模式")
        for task in tasks:
            process_single_video(task)

    elif mode == 2:
        max_processes = cpu_count()
        num_processes = int(input(f"输入进程数（最大 {max_processes}）："))
        num_processes = min(max_processes, num_processes)

        print(f"多进程模式，使用进程数: {num_processes}")
        with Pool(processes=num_processes) as pool:
            pool.map(process_single_video, tasks)


if __name__ == "__main__":
    real_video_dir = r"/home/inspur/STAR/dataSet/FaceForensics++/original_sequences/youtube/c23/videos"
    mask_video_dir = r"/home/inspur/STAR/dataSet/FaceForensics++/manipulated_sequences/Deepfakes/masks/videos"
    save_dir_base = r"/home/inspur/STAR/dataSet/FaceForensics++/image/REAL/original(128frames)/C23"
    target_size = 224
    desired_intervals = 128

    mode = int(input("选择模式（1：单进程，2：多进程）："))

    process_videos(real_video_dir, mask_video_dir, save_dir_base, mode,target_size,desired_intervals)