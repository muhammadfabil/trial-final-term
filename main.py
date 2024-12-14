import cv2
import mediapipe as mp
import random

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Daftar pertanyaan trivia
questions = [
    {
        "question": "Apa ibu kota Indonesia?",
        "options": ["A. Surabaya", "B. Jakarta", "C. Bandung", "D. Medan", "E. Bali"],
        "answer": 2  # Jakarta
    },
    {
        "question": "Siapa penemu bola lampu?",
        "options": ["A. Thomas Edison", "B. Isaac Newton", "C. Albert Einstein", "D. Nikola Tesla", "E. James Watt"],
        "answer": 1  # Thomas Edison
    },
    {
        "question": "Planet terbesar di tata surya?",
        "options": ["A. Bumi", "B. Mars", "C. Jupiter", "D. Saturnus", "E. Venus"],
        "answer": 3  # Jupiter
    },
    {
        "question": "Hewan tercepat di darat?",
        "options": ["A. Kuda", "B. Singa", "C. Harimau", "D. Cheetah", "E. Rusa"],
        "answer": 4  # Cheetah
    },
    {
        "question": "Lambang kimia air?",
        "options": ["A. O2", "B. CO2", "C. H2O", "D. NaCl", "E. H2SO4"],
        "answer": 3  # H2O
    },
    {
        "question": "Berapa jumlah provinsi di Indonesia per 2024?",
        "options": ["A. 33", "B. 34", "C. 37", "D. 38", "E. 39"],
        "answer": 4  # 38
    },
    {
        "question": "Siapa presiden pertama Indonesia?",
        "options": ["A. Soeharto", "B. Soekarno", "C. Habibie", "D. Megawati", "E. Gus Dur"],
        "answer": 2  # Soekarno
    },
    {
        "question": "Benua terkecil di dunia?",
        "options": ["A. Asia", "B. Afrika", "C. Eropa", "D. Australia", "E. Antartika"],
        "answer": 4  # Australia
    },
    {
        "question": "Apa hewan nasional Indonesia?",
        "options": ["A. Harimau Sumatra", "B. Elang Jawa", "C. Komodo", "D. Orangutan", "E. Gajah"],
        "answer": 3  # Komodo
    },
    {
        "question": "Apa bahasa resmi Perserikatan Bangsa-Bangsa (PBB)?",
        "options": ["A. Inggris", "B. Prancis", "C. Spanyol", "D. Arab", "E. Semua benar"],
        "answer": 5  # Semua benar
    }
]

# Fungsi untuk menghitung jumlah jari yang diangkat
def count_fingers(hand_landmarks):
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

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Pilih pertanyaan secara acak
current_question = random.choice(questions)
score = 0  # Skor pemain

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Tampilkan pertanyaan dan opsi di layar
    cv2.putText(frame, current_question["question"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    for i, option in enumerate(current_question["options"]):
        cv2.putText(frame, option, (10, 70 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Deteksi tangan dan hitung jari yang diangkat
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Gambar landmark tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Hitung jumlah jari yang diangkat
            fingers_count = count_fingers(hand_landmarks)

            # Tampilkan jawaban pemain
            if 1 <= fingers_count <= 5:  # Hanya valid untuk 1–5
                cv2.putText(frame, f"Jawaban Anda: {chr(64 + fingers_count)}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # Cek apakah jawaban benar
                if fingers_count == current_question["answer"]:
                    cv2.putText(frame, "Benar!", (10, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    score += 1
                    cv2.waitKey(2000)  # Tampilkan pesan selama 2 detik
                    current_question = random.choice(questions)  # Ganti ke pertanyaan baru

    # Tampilkan frame
    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    # Keluar jika menekan tombol 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()

print(f"Skor akhir Anda: {score}")
