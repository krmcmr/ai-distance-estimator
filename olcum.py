"""
A4 kagidini referans alarak webcam uzerinden nesne olcumu.

Calisma mantigi:
 1. best.pt modeli goruntudeki A4 kagidini bulur.
 2. Tespit kutusunun icinde kagidin 4 kosesi aranir. Bulunamazsa kutunun kendisi
    kullanilir (bu durumda olcum yaklasiktir, kamera kagida tam tepeden bakmali).
 3. Kagit, perspektif donusumuyle 210 x 297 mm'lik duz bir goruntuye cevrilir.
    Boylece her piksel sabit bir mm degerine karsilik gelir; kamera acisi olcumu bozmaz.
 4. Kagidin UZERINE konan (kagittan koyu ya da renkli) nesneler bulunur,
    uzunluk ve genislikleri mm cinsinden yazilir.

Kullanim:  python olcum.py [--model best.pt] [--kamera 0] [--conf 0.5]
Tuslar:    q = cikis, s = ekran goruntusu kaydet
"""
import argparse
import time

import cv2
import numpy as np

A4_KISA_MM, A4_UZUN_MM = 210.0, 297.0
PX_PER_MM = 3            # duzlestirilmis kagidin cozunurlugu
KENAR_BOSLUGU_MM = 6     # kagit kenarindaki golge ve kose hatalarini yok say
MIN_ALAN_MM2 = 150       # bundan kucuk lekeler nesne sayilmaz
KOYULUK_ESIGI = 45       # nesne, kagittan en az bu kadar koyu olmali (0-255)
DOYGUNLUK_ESIGI = 50     # ya da kagittan en az bu kadar renkli olmali (0-255)
YUMUSATMA = 0.6          # kose titremesini azaltir (0 = kapali, 1'e yaklastikca daha yumusak)


def koseleri_sirala(pts):
    """4 noktayi sol-ust, sag-ust, sag-alt, sol-alt sirasina koyar."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    toplam = pts.sum(axis=1)
    fark = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(toplam)], pts[np.argmin(fark)],
                     pts[np.argmax(toplam)], pts[np.argmax(fark)]], dtype=np.float32)


def kagit_koselerini_bul(frame, kutu):
    """YOLO kutusunun icinde kagidin gercek 4 kosesini arar. Bulamazsa None dondurur."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = kutu
    kutu_alani = (x2 - x1) * (y2 - y1)
    pad_x, pad_y = int((x2 - x1) * 0.1), int((y2 - y1) * 0.1)
    rx1, ry1 = max(0, int(x1) - pad_x), max(0, int(y1) - pad_y)
    rx2, ry2 = min(w, int(x2) + pad_x), min(h, int(y2) + pad_y)
    roi = frame[ry1:ry2, rx1:rx2]
    if roi.size == 0:
        return None

    gri = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, maske = cv2.threshold(gri, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not konturlar:
        return None

    # Kagidin uzerindeki nesneler kenara cikinti yapabilir; dis bukey zarf bunu kapatir
    zarf = cv2.convexHull(max(konturlar, key=cv2.contourArea))
    alan = cv2.contourArea(zarf)
    if not (0.5 * kutu_alani < alan < 1.3 * kutu_alani):
        return None  # arka plan da parlaksa kagit ayrilamaz
    yaklasik = cv2.approxPolyDP(zarf, 0.02 * cv2.arcLength(zarf, True), True)
    if len(yaklasik) != 4:
        return None
    return koseleri_sirala(yaklasik.reshape(4, 2) + [rx1, ry1])


def donusum_hesapla(koseler):
    """Kagit koselerinden, goruntuyu mm olcekli duz kagida ceviren matrisi hesaplar."""
    sol_ust, sag_ust, sag_alt, sol_alt = koseler
    genislik = np.linalg.norm(sag_ust - sol_ust) + np.linalg.norm(sag_alt - sol_alt)
    yukseklik = np.linalg.norm(sol_alt - sol_ust) + np.linalg.norm(sag_alt - sag_ust)
    w_mm, h_mm = (A4_UZUN_MM, A4_KISA_MM) if genislik >= yukseklik else (A4_KISA_MM, A4_UZUN_MM)
    W, H = int(w_mm * PX_PER_MM), int(h_mm * PX_PER_MM)
    hedef = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float32)
    return cv2.getPerspectiveTransform(koseler, hedef), (W, H)


def nesneleri_olc(duz):
    """Duzlestirilmis kagit uzerindeki nesneleri bulur: [(minAreaRect, uzunluk_mm, genislik_mm), ...]"""
    gri = cv2.GaussianBlur(cv2.cvtColor(duz, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    doygunluk = cv2.cvtColor(duz, cv2.COLOR_BGR2HSV)[:, :, 1]
    maske = ((gri < np.median(gri) - KOYULUK_ESIGI) |
             (doygunluk > np.median(doygunluk) + DOYGUNLUK_ESIGI)).astype(np.uint8) * 255

    b = KENAR_BOSLUGU_MM * PX_PER_MM
    maske[:b, :] = 0
    maske[-b:, :] = 0
    maske[:, :b] = 0
    maske[:, -b:] = 0
    maske = cv2.morphologyEx(maske, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    maske = cv2.morphologyEx(maske, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    sonuc = []
    konturlar, _ = cv2.findContours(maske, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for k in konturlar:
        if cv2.contourArea(k) < MIN_ALAN_MM2 * PX_PER_MM ** 2:
            continue
        rect = cv2.minAreaRect(k)
        w, h = rect[1]
        sonuc.append((rect, max(w, h) / PX_PER_MM, min(w, h) / PX_PER_MM))
    return sonuc


def yazi(img, metin, konum, renk=(255, 255, 255), olcek=0.55):
    x, y = int(konum[0]), int(konum[1])
    cv2.putText(img, metin, (x, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, metin, (x, y), cv2.FONT_HERSHEY_SIMPLEX, olcek, renk, 1, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description="A4 referansli webcam olcumu")
    ap.add_argument("--model", default="best.pt")
    ap.add_argument("--kamera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.5)
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    model = YOLO(args.model)
    cihaz = 0 if torch.cuda.is_available() else "cpu"
    print("Model siniflari:", model.names, "| cihaz:", cihaz)

    cap = cv2.VideoCapture(args.kamera, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    if not cap.isOpened():
        raise SystemExit(f"Kamera {args.kamera} acilamadi")

    onceki_koseler = None
    son_zaman = time.time()
    fps = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cizim = frame.copy()

        sonuc = model(frame, device=cihaz, conf=args.conf, verbose=False)[0]
        if len(sonuc.boxes) == 0:
            onceki_koseler = None
            yazi(cizim, "A4 kagit bulunamadi", (10, 30), (0, 0, 255), 0.7)
        else:
            en_iyi = int(sonuc.boxes.conf.argmax())
            kutu = sonuc.boxes.xyxy[en_iyi].cpu().numpy()
            guven = float(sonuc.boxes.conf[en_iyi])

            koseler = kagit_koselerini_bul(frame, kutu)
            kesin = koseler is not None
            if not kesin:
                x1, y1, x2, y2 = kutu
                koseler = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)

            # Kareden kareye titremeyi azalt; kagit hareket ettiyse hemen yeni konuma gec
            if onceki_koseler is not None and np.abs(koseler - onceki_koseler).mean() < 15:
                koseler = YUMUSATMA * onceki_koseler + (1 - YUMUSATMA) * koseler
            onceki_koseler = koseler

            M, boyut = donusum_hesapla(koseler)
            duz = cv2.warpPerspective(frame, M, boyut)
            M_ters = np.linalg.inv(M)

            kagit_rengi = (0, 255, 0) if kesin else (0, 200, 255)
            cv2.polylines(cizim, [koseler.astype(np.int32)], True, kagit_rengi, 2)
            durum = "kose tespiti" if kesin else "YAKLASIK (koseler bulunamadi)"
            yazi(cizim, f"A4 %{guven * 100:.0f} - {durum}", (10, 30), kagit_rengi, 0.6)

            for rect, uzunluk, genislik in nesneleri_olc(duz):
                kutu_duz = cv2.boxPoints(rect).reshape(-1, 1, 2).astype(np.float32)
                kutu_goruntu = cv2.perspectiveTransform(kutu_duz, M_ters).reshape(-1, 2)
                cv2.polylines(cizim, [kutu_goruntu.astype(np.int32)], True, (255, 0, 255), 2)
                cx, cy = kutu_goruntu.mean(axis=0)
                yazi(cizim, f"{uzunluk / 10:.1f} x {genislik / 10:.1f} cm", (cx - 50, cy))

                cv2.drawContours(duz, [cv2.boxPoints(rect).astype(np.int32)], 0, (255, 0, 255), 2)
                yazi(duz, f"{uzunluk:.0f} x {genislik:.0f} mm", rect[0], olcek=0.6)

            cv2.imshow("Duzlestirilmis A4", duz)

        simdi = time.time()
        fps = 0.9 * fps + 0.1 / max(simdi - son_zaman, 1e-6)
        son_zaman = simdi
        yazi(cizim, f"FPS {fps:.0f}", (10, cizim.shape[0] - 10))
        cv2.imshow("Olcum", cizim)

        tus = cv2.waitKey(1) & 0xFF
        if tus == ord("q"):
            break
        if tus == ord("s"):
            ad = time.strftime("olcum_%Y%m%d_%H%M%S.png")
            cv2.imwrite(ad, cizim)
            print("Kaydedildi:", ad)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
