import cv2
import mediapipe as mp
import joblib
import warnings
warnings.filterwarnings('ignore')

model = joblib.load('model.pkl')
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

etiketler = ['mutlu', 'uzgun', 'kizgin', 'saskin']
kamera = cv2.VideoCapture(0)

while True:
    ret, frame = kamera.read()
    if not ret:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    sonuc = face_mesh.process(rgb)

    if sonuc.multi_face_landmarks:
        for yuz in sonuc.multi_face_landmarks:
            koordinatlar = []
            for nokta in yuz.landmark:
                koordinatlar.extend([nokta.x, nokta.y])
            if len(koordinatlar) == model.n_features_in_:
                tahmin = model.predict([koordinatlar])[0]
                cv2.putText(frame, f"{tahmin}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow('Test', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

kamera.release()
cv2.destroyAllWindows()
