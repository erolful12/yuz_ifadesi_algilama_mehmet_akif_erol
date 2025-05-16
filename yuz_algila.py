# yuz_ifade_tanima.py – Tek Dosya: Veri Toplama, Eğitim, Test

import cv2
import csv
import mediapipe as mp
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Ortak
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
etiketler = ['mutlu', 'uzgun', 'kizgin', 'saskin']
veri_dosyasi = 'veriseti.csv'

# --- 1. Veri Toplama ---
while True:
    # Kamera üzerinde seçim yap
    kamera = cv2.VideoCapture(0)
    secim = None
    while secim is None:
        ret, frame = kamera.read()
        if not ret:
            break
        # Metinleri ekle
        cv2.putText(frame, "Hangi ifadeyi kaydetmek istiyorsunuz?", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        for i, et in enumerate(etiketler):
            cv2.putText(frame, f"{i}: {et}", (10, 60 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Cikmak icin q'ya basin", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Secim', frame)
        key = cv2.waitKey(1) & 0xFF
        if key in [ord(str(i)) for i in range(len(etiketler))]:
            secim = int(chr(key))
        elif key == ord('q'):
            secim = -1
    kamera.release()
    cv2.destroyWindow('Secim')
    if secim == -1:
        break
    etiket = etiketler[secim]

    # 300 ornek kaydet
    print(f"'{etiket}' kaydina baslandi...")
    kamera = cv2.VideoCapture(0)
    veriler, kayit_sayisi = [], 0
    while kayit_sayisi < 300:
        ret, frame = kamera.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        sonuc = face_mesh.process(rgb)
        if sonuc.multi_face_landmarks:
            for yuz in sonuc.multi_face_landmarks:
                coords = []
                for nokta in yuz.landmark:
                    coords.extend([nokta.x, nokta.y])
                coords.append(etiket)
                veriler.append(coords)
                kayit_sayisi += 1
        cv2.putText(frame, f"Kayit: {kayit_sayisi}/300", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.imshow('Veri Toplama', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    kamera.release()
    cv2.destroyWindow('Veri Toplama')

    if veriler:
        alan_sayisi = len(veriler[0]) - 1
        basliklar = [f"x{i//2}" if i%2==0 else f"y{i//2}" for i in range(alan_sayisi)] + ['etiket']
        dosya_var = os.path.exists(veri_dosyasi)
        with open(veri_dosyasi, 'a', newline='') as f:
            writer = csv.writer(f)
            if not dosya_var:
                writer.writerow(basliklar)
            writer.writerows(veriler)
        print(f"{len(veriler)} '{etiket}' kaydi '{veri_dosyasi}' dosyasina eklendi.")
    else:
        print("Kayit alinamadi.")

    # Devam secimi kamera uzerinde
    kamera = cv2.VideoCapture(0)
    cv2.namedWindow('Devam')
    devam = None
    while devam is None:
        ret, frame = kamera.read()
        if not ret:
            break
        cv2.putText(frame, "Baska ifade kaydetmek icin e, bitirmek icin h", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255),2)
        cv2.imshow('Devam', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('e'):
            devam = True
        elif key == ord('h'):
            devam = False
    kamera.release()
    cv2.destroyWindow('Devam')
    if not devam:
        break
print("Veri toplama tamamlandi.")
cv2.destroyAllWindows()
