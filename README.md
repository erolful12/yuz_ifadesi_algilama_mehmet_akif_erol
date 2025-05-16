Yüz İfadesi Tanıma Projesi - Çalıştırma Notları
Bu proje, Mediapipe ve Logistic Regression kullanarak gerçek zamanlı yüz ifadesi (mutlu, üzgün, kızgın, şaşkın) tanıma yapar.
Çalıştırma Notları
Gereksinimler

Python 3.8+
Kütüphaneler: pip install mediapipe opencv-python pandas scikit-learn joblib
Webcam
Mediapipe FaceMesh modeli (face_landmarker_v2_with_blendshapes.task)

Kurulum

Kütüphaneleri yükleyin: pip install mediapipe opencv-python pandas scikit-learn joblib
Proje dosyalarını (yuz_algila.py, egitim.py, yuz_algila_test.py) aynı dizine yerleştirin.
Webcam’in çalıştığından emin olun.

1. Veri Toplama (yuz_algila.py)

Çalıştırma: python yuz_algila.py
0-3 tuşlarıyla ifade seçin (0: mutlu, 1: üzgün, 2: kızgın, 3: şaşkın).
Her ifade için 300 örnek toplanır, ekranda kayıt sayısı gösterilir.
e ile devam, h ile bitir. Veriler veriseti.csv’ye kaydedilir.

2. Model Eğitimi (egitim.py)

Çalıştırma: python egitim.py
veriseti.csv kullanılarak model eğitilir (%96 doğruluk).
Eğitilmiş model model.pkl’ye kaydedilir.

3. Gerçek Zamanlı Test (yuz_algila_test.py)

Çalıştırma: python yuz_algila_test.py
Kamera açılır, ifadeler ekranda metin olarak gösterilir.
Çıkmak için q tuşuna basın.

Notlar

Veri toplarken farklı ışık ve açı kullanın.
Düşük FPS için kamera çözünürlüğünü düşürün.
Hata alırsanız, kütüphanelerin yüklü olduğunu ve kameranın çalıştığını kontrol edin.

