We have READ.md in both English and Persian
First Persian secend English

# 🧠 تحلیل رفتار مشتری بر اساس عوامل روانشناختی و فروش

> پروژه‌ای کامل از دریافت داده تا ساخت اپلیکیشن هوش مصنوعی — بر پایه بیش از **۵۵۰٬۰۰۰ رکورد** رفتار مشتری

---

## 📌 معرفی پروژه

در این پروژه یک دیتاست شبیه‌سازی‌شده با بیش از **۵۵۰٬۰۰۰ رکورد** از رفتار مشتریان ایجاد و تحلیل شده است. هدف اصلی بررسی ارتباط بین **ویژگی‌های روانشناختی افراد، رفتار دیجیتال، تعاملات بازاریابی و الگوی خرید** است.

این پروژه با رویکرد یک تحلیل‌گر داده انجام شده و شامل چرخه کامل یک پروژه Data Analytics است — از دریافت داده خام تا ساخت داشبورد و اپلیکیشن کاربردی مبتنی بر یادگیری ماشین.

---

## 🎯 سناریوی پروژه

فرض کنید یک شرکت فعال در حوزه فروش آنلاین محصولات مرتبط با توسعه فردی، سلامت روان و آموزش‌های شخصی‌سازی‌شده قصد دارد رفتار مشتریان خود را بهتر درک کند.

این شرکت می‌خواهد بداند:

- چه عوامل روانشناختی باعث افزایش خرید مشتریان می‌شوند؟
- چه گروه‌هایی از مشتریان ارزش بیشتری دارند؟
- چه عواملی باعث بازگشت مشتری و خرید مجدد می‌شوند؟
- چگونه می‌توان رفتار خرید آینده مشتریان را پیش‌بینی کرد؟

---

## 🎯 اهداف پروژه

- تحلیل رفتار خرید مشتریان
- بررسی تأثیر عوامل روانشناختی بر فروش
- شناسایی الگوهای پنهان در داده‌ها
- بررسی کیفیت داده‌ها و پاکسازی آن‌ها
- آماده‌سازی داده برای مدل‌های تحلیلی و یادگیری ماشین
- ساخت ابزار کاربردی برای نمایش نتایج تحلیل

---

## 📂 دیتاست

دیتاست شامل **۵۴۹٬۹۱۰ رکورد** و **۲۱ ویژگی** است و اطلاعات مشتریان را در چهار دسته پوشش می‌دهد:

### اطلاعات جمعیتی
| ستون | توضیح |
|------|--------|
| `age` | سن مشتری (۱۸ تا ۷۵ سال) |
| `gender` | جنسیت |
| `city` | شهر محل سکونت |
| `education_level` | سطح تحصیلات |
| `occupation` | شغل |

### ویژگی‌های روانشناختی
| ستون | نوع | توضیح |
|------|-----|--------|
| `stress_level` | منفی | سطح استرس (۰–۱۰) |
| `anxiety_score` | منفی | میزان اضطراب (۰–۱۰) |
| `self_esteem` | مثبت | اعتماد به نفس (۰–۱۰) |
| `optimism_score` | مثبت | خوش‌بینی (۰–۱۰) |
| `life_satisfaction` | مثبت | رضایت از زندگی (۰–۱۰) |
| `social_media_dependency` | منفی | وابستگی به شبکه‌های اجتماعی (۰–۱۰) |
| `impulsiveness` | منفی | میزان تکانشگری (۰–۱۰) |

### رفتار دیجیتال و بازاریابی
| ستون | توضیح |
|------|--------|
| `weekly_site_visits` | تعداد بازدید هفتگی از سایت |
| `campaign_exposure` | تعامل با کمپین‌های تبلیغاتی |

### اطلاعات فروش
| ستون | توضیح |
|------|--------|
| `total_orders` | تعداد کل سفارش‌ها |
| `average_order_value` | ارزش متوسط هر سفارش |
| `total_spent` | مجموع هزینه مشتری |
| `loyalty_points` | امتیاز وفاداری |
| `monthly_purchase_amount` | میزان خرید ماهانه |
| `discount_received` | میزان تخفیف دریافتی |

---

## 🗂️ ساختار پروژه

```
📦 project/
├── 📓 Understanding_and_Cleaning.ipynb   ← فاز ۱: درک و پاکسازی داده
├── 📓 i.ipynb                            ← فاز ۲: تحلیل و مدل‌سازی
├── 🐍 app.py                             ← اپلیکیشن Streamlit
├── 🐍 Loyalty_model.py                   ← موتور مدل (LoyaltyEngine)
├── 📁 classes/
│   ├── data_info_display.py              ← نمایش اطلاعات کلی دیتاست
│   ├── missing_values_analyzer.py        ← تحلیل و گزارش مقادیر گم‌شده
│   └── outlier_detector.py              ← تشخیص و مدیریت outlierها
├── 📁 models/                            ← مدل‌های ذخیره‌شده (joblib)
├── 📁 Report/                            ← گزارش‌های HTML/SVG خروجی
├── 📄 psychology_sales_dataset.csv       ← دیتاست خام
└── 📄 data_cleaned.csv                  ← دیتاست پاکسازی‌شده
```

---

## 🔄 مراحل پروژه

### فاز ۱ — درک و پاکسازی داده (`Understanding_and_Cleaning.ipynb`)

#### ۱.۱ بررسی اولیه داده
- نمایش اطلاعات کلی دیتاست با کلاس `DataInfoDisplay`
- بررسی انواع داده، مقادیر null و آمار توصیفی

#### ۱.۲ بررسی یکپارچگی داده (Sanity Check)
- تابع `check_aov_consistency` صحت ستون `average_order_value` را بررسی می‌کند
- فرمول: `AOV = total_spent / total_orders`
- ردیف‌هایی با اختلاف بیشتر از ۵٪ به عنوان mismatch شناسایی می‌شوند

#### ۱.۳ مدیریت مقادیر گم‌شده
با استفاده از کلاس `MissingValuesAnalyzer`:

| مرحله | روش |
|-------|-----|
| ردیف‌هایی با ≥۳ مقدار گم‌شده | حذف کامل ردیف |
| `anxiety_score` | رگرسیون خطی بر اساس `stress_level`, `optimism_score`, `life_satisfaction` |
| `self_esteem` | رگرسیون خطی بر اساس `stress_level`, `life_satisfaction`, `optimism_score` |
| مقادیر گم‌شده باقی‌مانده | پر کردن با median |

#### ۱.۴ مدیریت Outlierها
با استفاده از کلاس `OutlierDetector`:

| ستون | مشکل | راه‌حل |
|------|-------|---------|
| `weekly_site_visits` | skew = +11.37 | Box-Cox transform |
| `average_order_value` | skew = +23.49 | Box-Cox transform |
| `total_spent` | outlier جزئی | Winsorising (IQR) |
| `loyalty_points` | outlier جزئی | Winsorising (IQR) |
| `anxiety_score`, `age` | false positive | دست نخورده |

> نتیجه: تمام skewnessها به بازه (-1, +1) رسیدند.

#### ۱.۵ Feature Engineering
- `Age_Group`: گروه‌بندی سنی (۱۸–۲۵، ۲۶–۳۰، ...)
- `discount_rate`: نرخ تخفیف = `discount_received / total_spent` — شاخص حساسیت به قیمت

---

### فاز ۲ — تحلیل و مدل‌سازی (`i.ipynb`)

#### ۲.۱ آمار توصیفی
- تابع `show_descriptive_stats`: نمایش count, min, max, mean, median برای ویژگی‌های روانشناختی و مالی به صورت جداول رنگی Rich

#### ۲.۲ توزیع جمعیت‌شناختی
- تابع `show_value_counts`: توزیع Age_Group، جنسیت، شهر، تحصیلات، شغل

#### ۲.۳ سگمنت‌بندی مشتریان (RFM-style)
با تابع `calculate_rfm_scores` مشتریان در ۶ سگمنت طبقه‌بندی می‌شوند:

| سگمنت | رنگ | توضیح |
|--------|-----|--------|
| 🟢 Champions | سبز | خرید زیاد + هزینه بالا |
| 🟩 Loyal Customers | سبز روشن | خرید مکرر |
| 🔵 Big Spenders | آبی | هزینه بالا، خرید کمتر |
| 🟡 Potential Loyalists | زرد | پتانسیل رشد |
| ⚫ Regular Customers | خاکستری | رفتار متوسط |
| 🔴 At Risk / Low Value | قرمز | هزینه کم، خرید کم |

#### ۲.۴ پروفایل روانشناختی به تفکیک سگمنت
- تابع `show_psych_analysis`: مقایسه میانگین ویژگی‌های روانشناختی بین سگمنت‌ها
- رنگ‌بندی: سبز = بهتر، قرمز = بدتر (بر اساس نوع ویژگی)

#### ۲.۵ مدل‌سازی — LoyaltyEngine
مدل پیش‌بینی با **Gradient Boosting** آموزش دیده است:
- **هدف طبقه‌بندی**: پیش‌بینی سگمنت مشتری
- **هدف رگرسیون**: پیش‌بینی `total_spent`
- **فیچرها**: ۷ ویژگی روانشناختی
- خروجی‌ها: سگمنت، احتمال Champion، انتظار هزینه، احتمال همه سگمنت‌ها

---

### فاز ۳ — اپلیکیشن (`app.py`)

اپلیکیشن **Customer Loyalty Advisor** با Streamlit ساخته شده است:

#### ورودی
۷ اسلایدر در sidebar برای وارد کردن امتیاز روانشناختی مشتری (مقیاس ۰–۱۰):
- 😰 Stress Level · 😟 Anxiety Score · 💪 Self Esteem
- ⚡ Impulsiveness · 🌟 Optimism · 😊 Life Satisfaction · 📱 Social Media Dependency

#### خروجی‌ها

**متریک‌های کلیدی (۴ کارت):**
- Predicted Segment: سگمنت پیش‌بینی‌شده
- Champion Probability: احتمال تبدیل شدن به Champion
- Expected Total Spend: پیش‌بینی مجموع هزینه
- Large Gaps vs Champions: تعداد فاصله‌های قابل توجه با میانگین Champions

**نمودارها:**
- Radar Chart: مقایسه پروفایل روانشناختی مشتری با میانگین Champions
- Bar Chart: احتمال تعلق به هر سگمنت

**Feature Gap Analysis:**
- جدول فاصله هر ویژگی از میانگین Champions
- وضعیت: ✅ On track / ⚠️ Moderate gap / 🔴 Large concern

**Personalized Retention Recommendations:**
- استراتژی‌های شخصی‌سازی‌شده بر اساس نقاط ضعف مشتری
- مثال: اضطراب بالا → تأکید بر ضمانت بازگشت وجه، نظرات مشتریان، و پشتیبانی آسان

---

## 🧩 کلاس‌های کمکی (`classes/`)

### `DataInfoDisplay`
نمایش اطلاعات کلی دیتاست به صورت جداول Rich — shape، dtypes، null count، sample.

### `MissingValuesAnalyzer`
تحلیل کامل مقادیر گم‌شده:
- گزارش آماری کلی و به تفکیک ستون
- رنگ‌بندی بر اساس شدت: زرد (کم) → نارنجی (متوسط) → قرمز (زیاد)
- پیشنهادات علمی: MICE، KNN، median، regression imputation
- شناسایی ردیف‌هایی با ≥۳ مقدار گم‌شده به عنوان کاندیدای حذف
- export به CSV

### `OutlierDetector`
تشخیص، گزارش، تجسم و مدیریت outlierها:

| روش | کاربرد |
|-----|---------|
| IQR (Tukey Fences) | توزیع‌های متقارن |
| Z-Score | توزیع‌های نرمال |
| Modified Z-Score | توزیع‌های چوله (از MAD استفاده می‌کند) |
| Isolation Forest | outlierهای چندمتغیره |

ویژگی کلیدی: **تشخیص خودکار skewness** — اگر توزیع چوله باشد، هشدار می‌دهد که outlierهای شناسایی‌شده ممکن است false positive باشند.

---

## 🛠️ نصب و اجرا

### پیش‌نیازها

```bash
pip install pandas numpy scikit-learn scipy rich matplotlib seaborn streamlit plotly joblib
```

### اجرای notebook تحلیل

```bash
jupyter notebook Understanding_and_Cleaning.ipynb
jupyter notebook i.ipynb
```

### اجرای اپلیکیشن

```bash
streamlit run app.py
```

> **نکته:** قبل از اجرای اپلیکیشن، مدل باید train شده و در پوشه `models/` ذخیره شده باشد.

---

## 📊 نتایج کلیدی

- ویژگی‌های `optimism_score` و `self_esteem` بیشترین همبستگی مثبت را با خرید دارند
- مشتریان سگمنت **Champions** به طور میانگین اضطراب کمتر و خوش‌بینی بیشتری نسبت به سایر سگمنت‌ها دارند
- **Gradient Boosting** با ۷ ویژگی روانشناختی قادر است سگمنت مشتری را با دقت قابل قبول پیش‌بینی کند
- Box-Cox transform برای ستون‌هایی با skew بالای ۱۰ مؤثرتر از log-transform است

  ### میتوانید برای نمونه برنامه (به پی دی اف Customer Loyalty Advisor نگاه کنید.)


**✅ Here is the full English translation of your README:**

---

# 🧠 Customer Behavior Analysis Based on Psychological Factors and Sales

> A complete project from data ingestion to building an AI application — based on over **550,000 customer behavior records**

---

## 📌 Project Introduction

In this project, a simulated dataset with over **550,000 records** of customer behavior was created and analyzed. The main goal is to examine the relationship between **individuals’ psychological traits, digital behavior, marketing interactions, and purchasing patterns**.

The project was executed with a data analyst’s approach and covers the full cycle of a Data Analytics project — from raw data to building interactive dashboards and a machine learning-powered application.

---

## 🎯 Project Scenario

Imagine an online company selling products in the fields of personal development, mental health, and personalized training. The company wants to better understand its customers’ behavior.

They want to know:

- Which psychological factors drive increased customer purchases?
- Which customer groups are more valuable?
- What factors lead to customer retention and repeat purchases?
- How can we predict customers’ future buying behavior?

---

## 🎯 Project Objectives

- Analyze customer purchasing behavior
- Examine the impact of psychological factors on sales
- Identify hidden patterns in the data
- Assess data quality and perform cleaning
- Prepare data for analytical and machine learning models
- Build a practical tool to present analysis results

---

## 📂 Dataset

The dataset contains **549,910 records** and **21 features**, covering customer information in four main categories:

### Demographic Information
| Column              | Description                  |
|---------------------|------------------------------|
| `age`               | Customer age (18–75 years)   |
| `gender`            | Gender                       |
| `city`              | City of residence            |
| `education_level`   | Education level              |
| `occupation`        | Occupation                   |

### Psychological Features
| Column                    | Type     | Description                          |
|---------------------------|----------|--------------------------------------|
| `stress_level`            | Negative | Stress level (0–10)                  |
| `anxiety_score`           | Negative | Anxiety score (0–10)                 |
| `self_esteem`             | Positive | Self-esteem (0–10)                   |
| `optimism_score`          | Positive | Optimism score (0–10)                |
| `life_satisfaction`       | Positive | Life satisfaction (0–10)             |
| `social_media_dependency` | Negative | Social media dependency (0–10)       |
| `impulsiveness`           | Negative | Impulsiveness (0–10)                 |

### Digital Behavior & Marketing
| Column                | Description                              |
|-----------------------|------------------------------------------|
| `weekly_site_visits`  | Weekly website visits                    |
| `campaign_exposure`   | Exposure to advertising campaigns        |

### Sales Information
| Column                     | Description                        |
|----------------------------|------------------------------------|
| `total_orders`             | Total number of orders             |
| `average_order_value`      | Average value per order            |
| `total_spent`              | Total amount spent                 |
| `loyalty_points`           | Loyalty points                     |
| `monthly_purchase_amount`  | Monthly purchase amount            |
| `discount_received`        | Total discount received            |

---

## 🗂️ Project Structure

```
📦 project/
├── 📓 Understanding_and_Cleaning.ipynb   ← Phase 1: Data Understanding & Cleaning
├── 📓 i.ipynb                            ← Phase 2: Analysis & Modeling
├── 🐍 app.py                             ← Streamlit Application
├── 🐍 Loyalty_model.py                   ← Model Engine (LoyaltyEngine)
├── 📁 classes/
│   ├── data_info_display.py              ← Dataset overview display
│   ├── missing_values_analyzer.py        ← Missing values analysis
│   └── outlier_detector.py               ← Outlier detection & handling
├── 📁 models/                            ← Saved models (joblib)
├── 📁 Report/                            ← HTML/SVG reports
├── 📄 psychology_sales_dataset.csv       ← Raw dataset
└── 📄 data_cleaned.csv                   ← Cleaned dataset
```

---

## 🔄 Project Phases

### Phase 1 — Data Understanding & Cleaning (`Understanding_and_Cleaning.ipynb`)

#### 1.1 Initial Data Exploration
- Display general dataset information using `DataInfoDisplay` class
- Check data types, null values, and descriptive statistics

#### 1.2 Data Integrity Check (Sanity Check)
- Function `check_aov_consistency` validates `average_order_value`
- Formula: `AOV = total_spent / total_orders`
- Rows with >5% discrepancy are flagged as mismatches

#### 1.3 Missing Values Handling
Using `MissingValuesAnalyzer` class:

| Step                          | Method                                      |
|-------------------------------|---------------------------------------------|
| Rows with ≥3 missing values   | Complete row deletion                       |
| `anxiety_score`               | Linear regression (using stress, optimism, life satisfaction) |
| `self_esteem`                 | Linear regression (using stress, life satisfaction, optimism) |
| Remaining missing values      | Filled with median                          |

#### 1.4 Outlier Handling
Using `OutlierDetector` class:

| Column                  | Issue              | Solution                  |
|-------------------------|--------------------|---------------------------|
| `weekly_site_visits`    | Skew = +11.37      | Box-Cox transformation    |
| `average_order_value`   | Skew = +23.49      | Box-Cox transformation    |
| `total_spent`           | Minor outliers     | Winsorizing (IQR)         |
| `loyalty_points`        | Minor outliers     | Winsorizing (IQR)         |
| `anxiety_score`, `age`  | False positives    | Left untouched            |

> **Result:** All skewness reduced to the range (-1, +1).

#### 1.5 Feature Engineering
- `Age_Group`: Age grouping (18–25, 26–30, ...)
- `discount_rate`: Discount rate = `discount_received / total_spent` — price sensitivity indicator

---

### Phase 2 — Analysis & Modeling (`i.ipynb`)

#### 2.1 Descriptive Statistics
- Function `show_descriptive_stats`: Displays count, min, max, mean, median for psychological and financial features using colored Rich tables

#### 2.2 Demographic Distribution
- Function `show_value_counts`: Distribution of Age_Group, gender, city, education, and occupation

#### 2.3 Customer Segmentation (RFM-style)
Using `calculate_rfm_scores`, customers are divided into 6 segments:

| Segment                  | Color     | Description                     |
|--------------------------|-----------|---------------------------------|
| 🟢 Champions            | Green     | High purchase + high spend      |
| 🟩 Loyal Customers       | Light Green | Frequent buyers               |
| 🔵 Big Spenders          | Blue      | High spend, fewer purchases     |
| 🟡 Potential Loyalists   | Yellow    | Growth potential                |
| ⚫ Regular Customers     | Gray      | Average behavior                |
| 🔴 At Risk / Low Value   | Red       | Low spend and low frequency     |

#### 2.4 Psychological Profile by Segment
- Function `show_psych_analysis`: Comparison of average psychological traits across segments
- Color coding: Green = better, Red = worse (based on feature polarity)

#### 2.5 Modeling — LoyaltyEngine
A **Gradient Boosting** model was trained:
- **Classification target**: Predict customer segment
- **Regression target**: Predict `total_spent`
- **Features**: 7 psychological traits
- **Outputs**: Segment, Champion probability, expected spend, probability for all segments

---

### Phase 3 — Application (`app.py`)

**Customer Loyalty Advisor** — A Streamlit web application.

#### Input
7 sliders in the sidebar for psychological scores (0–10):
- 😰 Stress Level · 😟 Anxiety Score · 💪 Self Esteem
- ⚡ Impulsiveness · 🌟 Optimism · 😊 Life Satisfaction · 📱 Social Media Dependency

#### Outputs

**Key Metrics (4 Cards):**
- Predicted Segment
- Champion Probability
- Expected Total Spend
- Large Gaps vs Champions

**Visualizations:**
- Radar Chart: Customer’s psychological profile vs average Champions
- Bar Chart: Probability of belonging to each segment

**Feature Gap Analysis:**
- Table showing gaps for each trait compared to Champions
- Status: ✅ On track / ⚠️ Moderate gap / 🔴 Large concern

**Personalized Retention Recommendations:**
- Tailored strategies based on customer weaknesses
- Example: High anxiety → Emphasize money-back guarantee, customer testimonials, and easy support

---

## 🧩 Helper Classes (`classes/`)

### `DataInfoDisplay`
Displays general dataset information using Rich tables — shape, dtypes, null counts, sample rows.

### `MissingValuesAnalyzer`
Comprehensive missing values analysis:
- Overall and per-column statistical reports
- Color coding by severity: Yellow → Orange → Red
- Scientific suggestions (MICE, KNN, median, regression)
- Identifies rows with ≥3 missing values for deletion
- Export to CSV

### `OutlierDetector`
Detection, reporting, visualization, and handling of outliers using multiple methods:
- IQR, Z-Score, Modified Z-Score, Isolation Forest
- **Smart feature**: Automatically detects skewness and warns about possible false positives in skewed distributions

---

## 🛠️ Installation & Execution

### Prerequisites

```bash
pip install pandas numpy scikit-learn scipy rich matplotlib seaborn streamlit plotly joblib
```

### Running Notebooks

```bash
jupyter notebook Understanding_and_Cleaning.ipynb
jupyter notebook i.ipynb
```

### Running the Application

```bash
streamlit run app.py
```

> **Note:** The model must be trained and saved in the `models/` folder before running the app.

---

## 📊 Key Results

- `optimism_score` and `self_esteem` show the strongest positive correlation with purchasing behavior
- **Champions** segment has significantly lower anxiety and higher optimism compared to other segments
- **Gradient Boosting** model can predict customer segments with acceptable accuracy using only 7 psychological features
- Box-Cox transformation proved more effective than log-transform for columns with very high skewness (>10)

---

**Sample Application:**  
You can view the **Customer Loyalty Advisor** PDF for a visual example of the final application.

 
