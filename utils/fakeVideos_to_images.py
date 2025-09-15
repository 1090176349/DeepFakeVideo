import cv2
import os
import math
from tqdm import tqdm
from multiprocessing import cpu_count, Pool


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
    :param args: 参数列表，包含伪造视频路径、掩膜视频路径、保存路径、目标图像大小、期望采集帧数
    """
    fake_video_path, mask_video_path, save_dir, target_size, desired_intervals = args

    # 加载视频并检查帧数
    cap_fake, total_frames_fake = load_video(fake_video_path)
    cap_mask, total_frames_mask = load_video(mask_video_path)

    print(f"正在处理视频: {os.path.basename(fake_video_path)}，总帧数: {total_frames_fake}")

    # 动态计算帧间隔
    frame_interval = calculate_frame_interval(total_frames_fake, desired_intervals)
    frames_to_process = math.ceil(total_frames_fake / frame_interval)

    frame_index = 0
    frames_collected = 0

    # 使用 tqdm 显示进度条
    with tqdm(total=frames_to_process, desc=f"Processing {os.path.basename(fake_video_path)}", unit="frame") as pbar:
        while True:
            ret_fake, frame_fake = cap_fake.read()
            ret_mask, frame_mask = cap_mask.read()

            if not ret_fake or not ret_mask:
                break

            # 只处理指定间隔的帧
            if frame_index % frame_interval == 0:
                # 转为灰度图并创建二进制掩膜
                gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
                _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

                # 获取掩模区域的边界框
                x, y, w, h = cv2.boundingRect(binary_mask)

                expanded_x, expanded_y, expanded_w, expanded_h = expand_mask(frame_fake, x, y, w, h, target_size=target_size)

                # 从伪造帧中提取扩展区域
                face_img = frame_fake[expanded_y:expanded_y + expanded_h,
                                      expanded_x:expanded_x + expanded_w]

                save_face(face_img, save_dir, frame_index)
                frames_collected += 1  # 每采集一帧，更新已采集帧数

                # 更新进度条
                pbar.update(1)

            frame_index += 1

    # 处理完前面的帧后，如果采集的帧数小于 desired_intervals，则采集最后一帧
    if frames_collected < desired_intervals:
        # 跳到视频的最后一帧
        cap_fake.set(cv2.CAP_PROP_POS_FRAMES, total_frames_fake - 1)
        cap_mask.set(cv2.CAP_PROP_POS_FRAMES, total_frames_fake - 1)

        ret_fake, frame_fake = cap_fake.read()
        ret_mask, frame_mask = cap_mask.read()

        if ret_fake and ret_mask:
            # 转为灰度图并创建二进制掩膜
            gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
            _, binary_mask = cv2.threshold(gray_mask, 1, 255, cv2.THRESH_BINARY)

            # 获取掩模区域的边界框
            x, y, w, h = cv2.boundingRect(binary_mask)

            expanded_x, expanded_y, expanded_w, expanded_h = expand_mask(frame_fake, x, y, w, h, target_size=target_size)

            # 从伪造帧中提取扩展区域
            face_img = frame_fake[expanded_y:expanded_y + expanded_h,
                                  expanded_x:expanded_x + expanded_w]

            # 保存最后一帧
            save_face(face_img, save_dir, total_frames_fake - 1)
            frames_collected += 1  # 更新已采集的帧数

            # 更新进度条
            pbar.update(1)

    cap_fake.release()
    cap_mask.release()


def process_videos(fake_dir, mask_dir, save_dir_base, mode, target_size,desired_intervals):
    """
    处理目录下的所有视频文件
    :param fake_dir: 伪造视频路径
    :param mask_dir: 掩膜视频路径
    :param save_dir_base: 保存路径基目录
    :param save_size: 保存图像的大小
    :param mode: 处理模式，1 为单进程，2 为多进程
    """
    fake_files = sorted(os.listdir(fake_dir))
    mask_files = sorted(os.listdir(mask_dir))
    tasks = []

    for fake_file in fake_files:
        if fake_file in mask_files:
            fake_path = os.path.join(fake_dir, fake_file)
            mask_path = os.path.join(mask_dir, fake_file)
            save_dir = os.path.join(save_dir_base, os.path.splitext(fake_file)[0])
            tasks.append((fake_path, mask_path, save_dir,target_size,desired_intervals))

    if mode == 1:
        print("单进程模式")
        for task in tasks:
            process_single_video(task)

    elif mode == 2:
        max_processes = cpu_count()
        num_processes = int(input(f"输入进程数（最大 {max_processes})："))
        num_processes = min(max_processes, num_processes)

        print(f"多进程模式，使用进程数: {num_processes}")
        with Pool(processes=num_processes) as pool:
            pool.map(process_single_video, tasks)


if __name__ == "__main__":
    fake_video_dir = r"/home/inspur/STAR/dataSet/FaceForensics++/manipulated_sequences/NeuralTextures/c23/videos"
    mask_video_dir = r"/home/inspur/STAR/dataSet/FaceForensics++/manipulated_sequences/NeuralTextures/masks/videos"
    save_dir_base = r"/home/inspur/STAR/dataSet/FaceForensics++/image/FAKE/NeuralTextures/C23"

    target_size = 224
    desired_intervals=32
    mode = int(input("选择模式（1：单进程，2：多进程）："))

    process_videos(fake_video_dir, mask_video_dir, save_dir_base,mode,target_size,desired_intervals)