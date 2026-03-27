import requests, os
from PIL import Image
import numpy as np

API = "http://localhost:8088/api/analyze"
# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")

results = []

def test(path, label):
    try:
        with open(path, 'rb') as f:
            res = requests.post(API, files={'file': (os.path.basename(path), f, 'image/jpeg')})
            if res.status_code == 422:
                results.append(f"REJECTED  {label}")
            else:
                d = res.json()
                results.append(f"ACCEPTED  {label}  -> {d.get('prediction')} ({d.get('confidence',0):.3f})")
    except Exception as e:
        results.append(f"FAILED    {label}  -> {str(e)}")

# Generate OOD images (Temporary in local tests folder)
Image.new('RGB', (300,300), (100,160,230)).save("t1.jpg")    # blue sky
np_green = np.full((300,300,3), [34,120,50], dtype=np.uint8)
Image.fromarray(np_green).save("t2.jpg")                     # green forest
gray = np.random.randint(100,200,(300,300), dtype=np.uint8)
Image.fromarray(np.stack([gray]*3, axis=-1)).save("t3.jpg")   # grayscale
Image.new('RGB', (300,300), (220,40,40)).save("t4.jpg")       # red object

# Test OOD images
test("t1.jpg", "Blue Sky")
test("t2.jpg", "Green Forest") 
test("t3.jpg", "Grayscale")
test("t4.jpg", "Red Object")

# Test REAL skin images from data/uploads
if os.path.exists(UPLOADS_DIR):
    for f in os.listdir(UPLOADS_DIR):
        if f.endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(UPLOADS_DIR, f)
            try:
                # Check if it's not a completely flat/corrupt image natively first to label it
                std = np.array(Image.open(img_path).resize((128,128))).astype(float).std(axis=(0,1)).mean()
                label = f"Real Skin ({f[:8]}) [std={std:.1f}]"
                test(img_path, label)
            except Exception:
                pass

# Cleanup synthetic images
for f in ["t1.jpg","t2.jpg","t3.jpg","t4.jpg"]:
    if os.path.exists(f): os.remove(f)

results_path = os.path.join(BASE_DIR, "tests", "ood_results.txt")
with open(results_path, "w") as f:
    f.write("\n".join(results))
print(f"Done. Results written to {results_path}")
