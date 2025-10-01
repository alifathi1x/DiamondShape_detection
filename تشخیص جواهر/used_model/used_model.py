import os

import cv2
import numpy as np
from catboost import CatBoostClassifier, Pool

# -------------------- تنظیمات --------------------
model_path = r"C:\Users\Ali\PycharmProjects\PythonProject43\catboost_shape_model.cbm"
image_paths = [
    r"C:\Users\Ali\PycharmProjects\PythonProject43\images\1.jpg",
    r"C:\Users\Ali\PycharmProjects\PythonProject43\images\4.jpg",
    r"C:\Users\Ali\PycharmProjects\PythonProject43\images\5.jpg"
]

# -------------------- بارگذاری مدل --------------------
try:
    model = CatBoostClassifier()
    model.load_model(model_path)
    print("✅ مدل با موفقیت بارگذاری شد")
    print("تعداد ویژگی‌های مدل:", len(model.feature_names_))
    print("اسامی ویژگی‌ها:", model.feature_names_)
    cat_indices = model.get_cat_feature_indices()
    print("اندیس ویژگی‌های categorical:", cat_indices)
except Exception as e:
    print(f"❌ خطا در بارگذاری مدل: {e}")
    exit()

# -------------------- تعریف نام‌های تراش (Shape) --------------------
# بر اساس خروجی مدل، این نام‌ها را تعریف می‌کنیم
shape_names = {
    0: "Round",
    1: "Oval",
    2: "Pear",
    3: "Cushion Modified",
    4: "Emerald",
    5: "Princess",
    6: "Marquise",  # کلاس 6
    7: "Radiant",
    8: "Heart",
    9: "Asscher"  # کلاس 9
}

print("📋 نام‌های تراش تعریف شده:")
for class_id, shape_name in shape_names.items():
    print(f"  کلاس {class_id}: {shape_name}")


# -------------------- استخراج ویژگی از تصویر --------------------
def extract_features(image):
    height, width = image.shape[:2]

    # ترتیب EXACTLY مطابق با model.feature_names_
    features = [
        # [0] Cut - categorical
        "Ideal",
        # [1] Color - categorical
        "D",
        # [2] Clarity - categorical
        "VVS1",
        # [3] Carat Weight - numeric
        1.0,
        # [4] Length/Width Ratio - numeric
        float(width / height) if height != 0 else 1.0,
        # [5] Depth % - numeric
        60.0,
        # [6] Table % - numeric
        55.0,
        # [7] Polish - categorical
        "Excellent",
        # [8] Symmetry - categorical
        "Excellent",
        # [9] Girdle - categorical
        "Medium",
        # [10] Culet - categorical
        "None",
        # [11] Length - numeric
        float(width),
        # [12] Width - numeric
        float(height),
        # [13] Height - numeric
        float(height * 0.6),
        # [14] Price - numeric
        5000.0,
        # [15] Type - categorical
        "TypeA",
        # [16] Fluorescence - categorical
        "None"
    ]

    return np.array(features, dtype=object).reshape(1, -1)


# -------------------- پیش‌بینی و ترسیم --------------------
def predict_and_draw(image_path):
    # بررسی وجود فایل
    if not os.path.exists(image_path):
        print(f"❌ فایل {image_path} یافت نشد.")
        return True

    # بارگذاری تصویر
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ خطا در بارگذاری تصویر: {image_path}")
        return True

    print(f"\n📸 پردازش تصویر: {os.path.basename(image_path)}")
    print(f"📐 ابعاد تصویر: {img.shape[1]}x{img.shape[0]}")

    # استخراج ویژگی‌ها
    features = extract_features(img)

    try:
        # ساخت Pool برای پیش‌بینی
        data_pool = Pool(features, cat_features=cat_indices)

        # پیش‌بینی
        pred_class = model.predict(data_pool)

        # رفع خطای NumPy - استخراج صحیح مقدار از آرایه
        if hasattr(pred_class, 'flat'):
            pred_class_int = int(pred_class.flat[0])
        elif hasattr(pred_class, 'item'):
            pred_class_int = int(pred_class.item())
        else:
            pred_class_int = int(pred_class[0])

        print(f"🔮 کلاس پیش‌بینی شده (عدد): {pred_class_int}")

        # تبدیل به نام تراش
        if pred_class_int in shape_names:
            pred_label = shape_names[pred_class_int]
            print(f"💎 تراش پیش‌بینی شده: {pred_label}")
        else:
            pred_label = f"Class_{pred_class_int}"
            print(f"⚠️ کلاس ناشناخته: {pred_label}")

        # همچنین احتمال‌ها را نیز دریافت می‌کنیم
        pred_proba = model.predict_proba(data_pool)
        print(f"📊 احتمالات: {pred_proba[0]}")

    except Exception as e:
        print(f"❌ خطا در پیش‌بینی: {e}")
        return True

    # رسم روی تصویر
    height, width, _ = img.shape

    # مستطیل دور تصویر
    cv2.rectangle(img, (10, 10), (width - 10, height - 10), (0, 255, 0), 3)

    # متن پیش‌بینی
    cv2.putText(img, f"Shape: {pred_label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # اطلاعات اضافی
    cv2.putText(img, f"Size: {width}x{height}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(img, f"Class: {pred_class_int}", (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # نمایش تصویر
    window_name = f"Prediction: {pred_label}"
    cv2.imshow(window_name, img)

    # انتظار برای کلید
    print("⏳ منتظر فشار کلید... (Space=بعدی, Q=خروج)")
    key = cv2.waitKey(0)

    # بستن پنجره
    cv2.destroyAllWindows()

    # اگر کاربر دکمه 'q' یا 'Q' را فشار داد، برنامه متوقف شود
    if key in [ord('q'), ord('Q')]:
        return False

    return True


# -------------------- اجرای اصلی --------------------
def main():
    print("\n" + "=" * 60)
    print("🚀 سیستم تشخیص شکل الماس - CatBoost Model")
    print("=" * 60)

    # بررسی وجود فایل‌های تصویر
    valid_images = []
    for path in image_paths:
        if os.path.exists(path):
            valid_images.append(path)
        else:
            print(f"⚠️ فایل {path} وجود ندارد")

    if not valid_images:
        print("❌ هیچ فایل تصویری یافت نشد!")
        return

    print(f"📁 تعداد تصاویر قابل پردازش: {len(valid_images)}")
    print("\nدستورالعمل:")
    print("  - Space/Enter: نمایش تصویر بعدی")
    print("  - Q: خروج از برنامه")
    print("-" * 60)

    # پردازش تصاویر
    for i, path in enumerate(valid_images, 1):
        print(f"\n📍 تصویر {i} از {len(valid_images)}")

        continue_processing = predict_and_draw(path)

        if not continue_processing:
            print("\n⏹️ پردازش توسط کاربر متوقف شد")
            break

    print("\n" + "=" * 60)
    print("✅ پردازش تمام شد")
    print("🙏 از استفاده از برنامه متشکریم!")
    print("=" * 60)


# -------------------- اجرای برنامه --------------------
if __name__ == "__main__":
    main()
