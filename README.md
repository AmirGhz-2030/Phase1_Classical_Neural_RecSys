# 🎬 پروژه فاز ۱: مبانی سیستم‌های توصیه‌گر کلاسیک و عصبی (Phase 1 RecSys)

## 📌 مشخصات پروژه
* **موضوع پژوهش:** سیستم‌های توصیه‌گر عامل‌محور نسل جدید (Next-Generation Agentic Recommender Systems)
* **فاز جاری:** **فاز ۱ — مبانی و بنچمارک سیستم‌های توصیه‌گر کلاسیک و عصبی**
* **توسعه‌دهنده:** [AmirGhz-2030](https://github.com/AmirGhz-2030)
* **دیتاست مرجع:** MovieLens 20M Dataset (شامل ۲۰ میلیون تعامل کاربر-فیلم)

---

## 🎯 اهداف فاز ۱
این پروژه دقیقاً مطابق با سرفصل‌های **Phase 1: Fundamentals of Recommender Systems** پیاده‌سازی شده است و اهداف زیر را محقق می‌کند:
1. پیاده‌سازی و استقرار کامل خط لوله پیش‌پردازش داده (Data Pipeline) با اعمال فیلتر $20$-Core و تفکیک زمانی (Temporal Leave-One-Last).
2. پیاده‌سازی مدل‌های پایه‌ای سنتی و کلاسیک:
   - **Popularity Baseline:** توصیه‌گر بر مبنای فراوانی و محبوبیت سراسری.
   - **Item-KNN (CF):** فیلترینگ مشارکتی مبتنی بر آیتم با شباهت کسینوسی (Cosine Similarity).
   - **Matrix Factorization (SVD):** فاکتورگیری ماتریسی در فضای ۶۴ بُعدی متغیرهای پنهان (Latent Factors).
3. پیاده‌سازی مدل یادگیری عمیق **Neural Collaborative Filtering (NCF / NeuMF)** در پایتورچ با تلفیق شاخه‌های GMF و MLP.
4. پیاده‌سازی موتور ارزیابی رتبه‌بندی استاندارد آکادمیک با ۵ سنجه: $HR@K$، $NDCG@K$، $MRR@K$، $Precision@K$ و $Recall@K$.
5. ارزیابی تطبیقی در دو پروتکل علمی: **All-Item Ranking (۱۳,۱۳۰ آیتم)** و **Sampled-100 (پروتکل مقاله NCF 2017)**.
6. ساخت داشبورد وب تعاملی و پروداکشن با **Streamlit** و **Plotly** برای ارائه زنده و تست بلادرنگ.

---

## 🏗️ معماری ماژولار سیستم (System Architecture)

ساختار پوشه‌بندی پروژه بر اساس اصل **Separation of Concerns** به شکل زیر سازمان‌دهی شده است:

```text
Phase1_Classical_Neural_RecSys/
├── .gitignore                   # جلوگیری از کامیت فایل‌های حجیم دیتاست و محیط مجازی
├── README.md                    # مستندات کامل فنی و آموزشی پروژه
├── requirements.txt             # کتابخانه‌های پروژه (pandas, torch, streamlit, plotly, ...)
├── app/                         # ماژول رابط کاربری وب
│   ├── __init__.py
│   └── main.py                  # اپلیکیشن تعاملی Streamlit با ۴ صفحه مجزا
├── data/
│   ├── raw/                     # فایل‌های خام MovieLens 20M
│   └── processed/               # دیتاست‌های پالایش‌شده (train.csv, val.csv, test.csv, meta.pkl)
├── src/                         # هسته اصلی کدهای پایتون
│   ├── data/
│   │   ├── __init__.py
│   │   └── preprocessor.py      # پایپ‌لاین فیلتر K-Core و تفکیک زمانی
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classical.py         # پیاده‌سازی مدل‌های Popularity, Item-KNN, SVD
│   │   └── ncf.py               # پیاده‌سازی مدل عمیق NeuMF در PyTorch
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py           # فرمول‌های ریاضی و کلاس RankingEvaluator
│   │   ├── benchmark_classical.py # اسکریپت بنچمارک مدل‌های کلاسیک
│   │   ├── benchmark_ncf.py       # اسکریپت بنچمارک NCF
│   │   └── benchmark_sampled100.py# اسکریپت بنچمارک پروتکل ۱۰۰تایی
│   └── utils/
└── tests/                       # تست‌های صحت داده و تست‌های واحد سنجه‌ها
    ├── test_data_integrity.py
    └── test_metrics.py
```

---

## 📊 نتایج آزمایش‌ها و جدول لیدربورد بنچمارک

### ۱. ارزیابی در پروتکل Sampled-100 (مشابه مقاله NCF 2017)
*(برای هر کاربر تست: ۱ فیلم هدف + ۹۹ فیلم منفی تصادفی)*

| رتبه | مدل توصیه‌گر | $HR@5$ | $HR@10$ | $HR@20$ | $NDCG@10$ | $MRR@10$ |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Matrix Factorization (SVD)** | **۷۰.۵۶%** | **۸۰.۹۸%** | **۸۸.۱۴%** | **۰.۵۸۱۰** | **۰.۵۰۸۴** |
| 🥈 | **Item-KNN (CF)** | ۶۲.۲۲% | ۷۷.۱۸% | ۸۹.۳۸% | ۰.۵۱۰۹ | ۰.۴۲۹۶ |
| 🥉 | **Popularity** | ۵۲.۶۰% | ۶۸.۴۰% | ۸۳.۶۰% | ۰.۴۲۹۳ | ۰.۳۵۰۶ |

---

### ۲. ارزیابی در پروتکل All-Item Ranking (سخت‌گیرانه روی کل ۱۳,۱۳۰ فیلم)
*(توصیه از میان تمام آیتم‌های کاتالوگ — رفرنس مقالات ACM KDD)*

| رتبه | مدل توصیه‌گر | $HR@5$ | $HR@10$ | $HR@20$ | $NDCG@10$ | $MRR@10$ | زمان آموزش |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **Matrix Factorization (SVD)** | **۵.۴۰%** | **۹.۰۶%** | **۱۴.۹۲%** | **۰.۰۴۶۴** | **۰.۰۳۳۰** | ۹.۲ ثانیه |
| 🥈 | **Item-KNN (CF)** | ۳.۹۶% | ۶.۹۶% | ۱۱.۶۴% | ۰.۰۳۴۷ | ۰.۰۲۴۳ | ۵.۹ ثانیه |
| 🥉 | **Popularity** | ۳.۰۰% | ۵.۱۲% | ۸.۳۸% | ۰.۰۲۵۵ | ۰.۰۱۷۸ | ۴.۲ ثانیه |
| ۴ | **NCF (NeuMF)** | ۲.۷۴% | ۴.۹۲% | ۷.۹۸% | ۰.۰۲۴۴ | ۰.۰۱۷۰ | ۵۴۸ ثانیه |

> 💡 **نکته علمی:** حدس تصادفی در فضای ۱۳,۱۳۰ فیلم معادل **۰.۰۷۶٪** است. مدل SVD پیاده‌شده با رسیدن به $HR@10 = 9.06\%$، دقتی **۱۱۹ برابر بهتر از حدس تصادفی** به ثبت رسانده است.

---

## 🚀 راهنمای راه‌اندازی و اجرای پروژه (Quick Start)

### ۱. فعال‌سازی محیط مجازی پایتون
```bash
# در سیستم‌عامل ویندوز (CMD / PowerShell):
.\.venv\Scripts\activate

# یا در Git Bash:
source .venv/Scripts/activate
```

### ۲. اجرای پایپ‌لاین پیش‌پردازش داده‌ها
```bash
python src/data/preprocessor.py
```

### ۳. اجرای تست‌های واحد و راستی‌آزمایی
```bash
python tests/test_metrics.py
```

### ۴. اجرای بنچمارک‌ها
```bash
# بنچمارک مدل‌های کلاسیک:
python src/evaluation/benchmark_classical.py

# بنچمارک پروتکل ۱۰۰تایی:
python src/evaluation/benchmark_sampled100.py
```

### ۵. اجرای داشبورد تعاملی Streamlit
```bash
streamlit run app/main.py
```
سپس مرورگر خود را باز کرده و به آدرس زیر بروید:
👉 **`http://localhost:8501`**

---

## 👨‍💻 GitHub
* Profile: [@AmirGhz-2030](https://github.com/AmirGhz-2030)
* Repository: `Phase1_Classical_Neural_RecSys`
