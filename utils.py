# utils.py

import cv2
from PIL import Image
import numpy as np

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

def load_question_image(image_path, frame, width=300, height=200, y_offset=50):
    """
    Memuat dan menempatkan gambar pertanyaan ke dalam frame, mendukung transparansi.
    """
    question_image = Image.open(image_path).convert("RGBA")
    question_image = question_image.resize((width, height))

    # Konversi ke format yang kompatibel dengan OpenCV
    question_image_np = np.array(question_image)
    bgr_image = question_image_np[..., :3]  # Kanal BGR
    alpha_channel = question_image_np[..., 3]  # Kanal Alfa

    # Hitung posisi tengah atas
    frame_height, frame_width = frame.shape[:2]
    x_offset = (frame_width - width) // 2
    y_offset = y_offset

    # Tempelkan gambar dengan transparansi
    for c in range(0, 3):  # Untuk setiap kanal warna (BGR)
        frame_slice = frame[y_offset:y_offset + height, x_offset:x_offset + width, c]
        frame[y_offset:y_offset + height, x_offset:x_offset + width, c] = \
            frame_slice * (1 - alpha_channel / 255.0) + bgr_image[..., c] * (alpha_channel / 255.0)

    # Ubah ukuran gambar agar sesuai dengan frame yang akan diupdate
    question_image_resized = cv2.resize(bgr_image, (width, height))

    return question_image_resized, (x_offset, y_offset, width, height)


def play_gif(gif_path, frame, position):
    """
    Memutar animasi GIF pada posisi tertentu dalam frame.
    """
    x, y, w, h = position
    gif = Image.open(gif_path)

    for frame_gif in range(gif.n_frames):
        gif.seek(frame_gif)
        gif_frame = gif.convert("RGBA")
        gif_frame = gif_frame.resize((w, h))

        # Konversi ke format OpenCV
        gif_np = np.array(gif_frame)
        bgr_frame = gif_np[..., :3]
        alpha_channel = gif_np[..., 3]

        # Gabungkan GIF ke frame utama
        for c in range(0, 3):
            frame_slice = frame[y:y + h, x:x + w, c]
            frame[y:y + h, x:x + w, c] = \
                frame_slice * (1 - alpha_channel / 255.0) + bgr_frame[..., c] * (alpha_channel / 255.0)

        # Tampilkan frame GIF
        cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)
        cv2.waitKey(100)  # Sesuaikan kecepatan animasi GIF
