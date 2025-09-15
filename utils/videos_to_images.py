import os
import cv2
import dlib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count  # 用于多进程处理

# 加载 dlib 的 HOG 人脸检测器
hog_face_detector = dlib.get_frontal_face_detector()

def extract_faces_from_video(video_path, output_dir, target_frame_count, target_size=(100, 100)):
    """
    从视频中提取最大面积的人脸，按目标帧数进行抽取，并将人脸图像保存到指定目录。
    :param video_path: 视频文件路径
    :param output_dir: 人脸图像保存的目标目录
    :param target_frame_count: 从视频中提取的总帧数
    :param target_size: 保存人脸的尺寸 (width, height)
    """
    # 确保输出目录存在
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频文件: {video_path}")
        return
    
    # 获取视频的帧数
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 确定每帧的间隔
    frame_interval = total_frames // target_frame_count
    
    # 获取视频名称，用于创建输出文件夹
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    Path(video_output_dir).mkdir(parents=True, exist_ok=True)
    
    # 读取并处理帧，显示进度条
    for i in tqdm(range(target_frame_count), desc=f"处理视频 {video_name}", unit="帧"):
        # 设置当前读取的帧位置
        frame_num = i * frame_interval
        
        # 设置视频的当前帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # 读取当前帧
        ret, frame = cap.read()
        if not ret:
            break
        
        # 转为灰度图像
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 使用 HOG 人脸检测器检测
        faces = hog_face_detector(gray, 1)
        
        # 选择面积最大的人脸
        max_face = None
        max_area = 0
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            area = w * h
            if area > max_area:
                max_area = area
                max_face = (x, y, w, h)
        
        if max_face:
            x, y, w, h = max_face
            
            # 提取最大人脸区域
            face_image = frame[y:y+h, x:x+w]
            
            # 调整人脸图像的大小
            resized_face = cv2.resize(face_image, target_size)
            
            # 构造保存路径
            face_filename = f"frame_{frame_num}_face.jpg"
            face_save_path = os.path.join(video_output_dir, face_filename)
            
            # 保存最大人脸图像
            cv2.imwrite(face_save_path, resized_face)
            print(f"保存最大人脸图像: {face_save_path}")
    
    # 释放视频文件
    cap.release()

def process_videos_in_folder(input_folder, output_folder, target_frame_count, target_size=(100, 100), use_multiprocessing=False, num_processes=None):
    """
    遍历文件夹中的所有视频文件，并对每个视频进行人脸提取。
    :param input_folder: 视频文件所在的文件夹
    :param output_folder: 人脸图像保存的目标文件夹
    :param target_frame_count: 从每个视频中提取的总帧数
    :param target_size: 保存人脸的尺寸 (width, height)
    :param use_multiprocessing: 是否使用多进程
    :param num_processes: 使用的进程数
    """
    # 获取文件夹中的所有视频文件
    video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mov'))]
    
    # 单进程模式
    if not use_multiprocessing:
        for video_file in video_files:
            video_path = os.path.join(input_folder, video_file)
            print(f"处理视频: {video_path}")
            extract_faces_from_video(video_path, output_folder, target_frame_count, target_size)
    
    # 多进程模式
    else:
        # 创建进程池，并根据用户选择的进程数进行处理
        with Pool(processes=num_processes) as pool:
            pool.starmap(extract_faces_from_video, [(os.path.join(input_folder, video_file), output_folder, target_frame_count, target_size) for video_file in video_files])

if __name__ == '__main__':
    input_folder = r"/home/inspur/STAR/dataSet/FaceForensics++/original_sequences/youtube/c23/videos"  # 替换为视频文件夹路径
    output_folder = r"/home/inspur/STAR/dataSet/FaceForensics++/image/REAL/original(128frames)/C23"  # 替换为保存人脸图像的输出文件夹路径
    target_frame_count = 128  # 设置从每个视频中提取的帧数
    target_size = (224, 224)  # 设置保存的人脸图像尺寸

    # 用户输入选择模式
    mode = input("请输入模式：1 - 单进程模式，2 - 多进程模式: ").strip()

    if mode == "1":
        print("使用单进程模式处理视频...")
        process_videos_in_folder(input_folder, output_folder, target_frame_count, target_size, use_multiprocessing=False)
    elif mode == "2":
        # 获取最大可用进程数
        max_processes = cpu_count()
        print(f"最大可用进程数为：{max_processes}")
        
        # 用户输入要使用的进程数
        num_processes = input(f"请输入要使用的进程数 (最大值为 {max_processes}): ").strip()
        
        try:
            num_processes = int(num_processes)
            if num_processes < 1 or num_processes > max_processes:
                print(f"无效的进程数，必须在 1 到 {max_processes} 之间。")
            else:
                print(f"使用 {num_processes} 个进程处理视频...")
                process_videos_in_folder(input_folder, output_folder, target_frame_count, target_size, use_multiprocessing=True, num_processes=num_processes)
        except ValueError:
            print("无效的输入，请输入一个整数。")
    else:
        print("无效的输入，请选择 1 或 2。")
