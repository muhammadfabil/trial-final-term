# utils.py

import cv2
from PIL import Image

def count_fingers(hand_landmarks):
    """
    Menghitung jumlah jari yang diangkat berdasarkan landmark tangan.
    """
    finger_tips = [8, 12, 16, 20]  # Ujung jari
    finger_pips = [6, 10, 14, 18]  # Sendi bawah ujung jari

    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:  # Jika ujung jari lebih tinggi
            count += 1

    # Periksa ibu jari
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.1:  # Threshold untuk ibu jari
        count += 1

    return count

import cv2
import numpy as np

def load_question_image(image_path, frame, width=300, height=200, y_offset=50):
    """
    Memuat dan menempatkan gambar pertanyaan ke dalam frame, mendukung transparansi.
    """
    # Memuat gambar menggunakan PIL untuk mendukung transparansi
    question_image = Image.open(image_path).convert("RGBA")
    question_image = question_image.resize((width, height))

    # Konversi ke format numpy array
    question_image_np = np.array(question_image)

    # Pisahkan saluran RGBA
    r, g, b, a = cv2.split(question_image_np)

    # Gabungkan kembali menjadi gambar dengan transparansi
    bgr_image = cv2.merge([b, g, r])  # RGB to BGR
    alpha_channel = a

    # Hitung posisi tengah atas
    x_offset = (frame.shape[1] - width) // 2

    # Tempelkan gambar dengan transparansi
    for c in range(0, 3):  # Untuk setiap kanal warna (BGR)
        frame_slice = frame[y_offset:y_offset + height, x_offset:x_offset + width, c]
        frame[y_offset:y_offset + height, x_offset:x_offset + width, c] = \
            frame_slice * (1 - alpha_channel / 255.0) + bgr_image[..., c] * (alpha_channel / 255.0)

    return bgr_image, (x_offset, y_offset, width, height)   

