import requests
import random
import threading
import time
import psutil
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes
import logging
import re
import json
from geopy.geocoders import Nominatim
from urllib.parse import urlparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from itertools import cycle
from bs4 import BeautifulSoup  # لتحليل محتوى الصفحة
import optuna  # لتحسين النموذج

# **1. بيانات الاعتماد والوكلاء**
USERNAME = os.environ.get('PROXY_USERNAME', 'Yahia_FwOsV')
PASSWORD = os.environ.get('PROXY_PASSWORD', 'Yahia+14118482')
COUNTRY = 'US'
def create_proxy_url(username, password, country):
    return f'http://customer-{username}-cc-{country}:{password}@pr.oxylabs.io:7777'
PROXY_URL = create_proxy_url(USERNAME, PASSWORD, COUNTRY)

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8121636489:AAGMJcwUhHi-Bk0TDhSrIvASNPaSoF4XZjk')

# **2. الإعدادات الافتراضية**
DEFAULT_CONFIG = {
    "DEFAULT_VISITS_PER_DAY": 7000,
    "DEFAULT_CONCURRENCY": 75,
    "MAX_CONCURRENCY": 250,
    "MIN_CONCURRENCY": 15,
    "MAX_RETRIES": 5,
    "PROXY_ROTATION_ENABLED": True,
    "LOG_LEVEL": "INFO",
    "TARGET_URL": "https://ip.oxylabs.io/location",
    "USER_AGENT_ROTATION_ENABLED": True,
    "SMART_ERROR_HANDLING": True,
    "AI_ENABLED": True,
    "AI_ADJUST_CONCURRENCY": True,
    "AI_ADJUST_VISITS": True,
    "AI_MONITOR_INTERVAL": 300,
    "SUCCESS_RATE_THRESHOLD": 0.85,
    "FAILURE_RATE_THRESHOLD": 0.15,
    "RESPONSE_TIME_THRESHOLD": 4,
    "BOUNCE_RATE_THRESHOLD": 0.4,
    "PROXY_ROTATION_FREQUENCY": 0.7,
    "CONCURRENCY_ADJUSTMENT_FACTOR": 0.1,
    "VISITS_ADJUSTMENT_FACTOR": 0.05,
    "PROXY_VALIDATION_THRESHOLD": 0.8,
    "DATA_COLLECTION_ENABLED": True,
    "DATA_COLLECTION_INTERVAL": 3600,
    "ANONYMOUS_PROXIES_ENABLED": True,
    "OPTIMIZE_MODEL_ENABLED": True, # تفعيل/تعطيل تحسين النموذج التلقائي
    "NUM_TRIALS": 5, # عدد التجارب لتحسين النموذج
}

# **3. تحميل التكوين من ملف**
def load_config(filename="config.json"):
    try:
        with open(filename, "r") as f:
            config = json.load(f)
            return {**DEFAULT_CONFIG, **config}
    except FileNotFoundError:
        print("Config file not found, using default config.")
        return DEFAULT_CONFIG

CONFIG = load_config()

# **4. المتغيرات العالمية**
visits_count = 0
successful_visits = 0
failed_visits = 0
running = False
bot_running = False
lock = threading.Lock()
target_url = CONFIG["TARGET_URL"]
current_concurrency = CONFIG["DEFAULT_CONCURRENCY"]
current_visits_per_day = CONFIG["DEFAULT_VISITS_PER_DAY"]
proxy_rotation_enabled = CONFIG["PROXY_ROTATION_ENABLED"]
smart_error_handling = CONFIG["SMART_ERROR_HANDLING"]
ai_enabled = CONFIG["AI_ENABLED"]
ai_adjust_concurrency = CONFIG["AI_ADJUST_CONCURRENCY"]
ai_adjust_visits = CONFIG["AI_ADJUST_VISITS"]
ai_monitor_interval = CONFIG["AI_MONITOR_INTERVAL"]
success_rate_threshold = CONFIG["SUCCESS_RATE_THRESHOLD"]
failure_rate_threshold = CONFIG["FAILURE_RATE_THRESHOLD"]
response_time_threshold = CONFIG["RESPONSE_TIME_THRESHOLD"]
bounce_rate_threshold = CONFIG["BOUNCE_RATE_THRESHOLD"]
proxy_rotation_frequency = CONFIG["PROXY_ROTATION_FREQUENCY"]
concurrency_adjustment_factor = CONFIG["CONCURRENCY_ADJUSTMENT_FACTOR"]
visits_adjustment_factor = CONFIG["VISITS_ADJUSTMENT_FACTOR"]
proxy_validation_threshold = CONFIG["PROXY_VALIDATION_THRESHOLD"]
data_collection_enabled = CONFIG["DATA_COLLECTION_ENABLED"]
data_collection_interval = CONFIG["DATA_COLLECTION_INTERVAL"]
anonymous_proxies_enabled = CONFIG["ANONYMOUS_PROXIES_ENABLED"]
optimize_model_enabled = CONFIG["OPTIMIZE_MODEL_ENABLED"]
num_trials = CONFIG["NUM_TRIALS"]

# **5. قائمة وكيل المستخدم**
user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36 Edg/91.0.864.59",
]
user_agent_cycle = cycle(user_agents) # تدوير وكلاء المستخدم

# **6. قائمة الوكلاء**
proxy_list = [
    PROXY_URL,
]
current_proxy_index = 0
proxy_cycle = cycle(proxy_list) # تدوير الوكلاء

# **7. نموذج التعلم متعدد المهام (Multi-Task Learning)**
class MultiTaskModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_tasks):
        super(MultiTaskModel, self).__init__()
        self.shared_layer = nn.Linear(input_size, hidden_size)
        self.task_layers = nn.ModuleList([nn.Linear(hidden_size, 1) for _ in range(num_tasks)])

    def forward(self, x):
        x = torch.relu(self.shared_layer(x))
        outputs = [torch.sigmoid(task_layers(x)) for task_layers in self.task_layers]
        return outputs

# **8. تهيئة نموذج التعلم متعدد المهام**
input_size = 10  # حجم المدخلات (تم توسيعه)
hidden_size = 32  # زيادة عدد الوحدات المخفية
num_tasks = 2  # عدد المهام (التنبؤ بالأخطاء، التحقق من صحة الوكيل)
model = MultiTaskModel(input_size, hidden_size, num_tasks)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# **9. تهيئة المسجل**
log_level = getattr(logging, CONFIG["LOG_LEVEL"].upper(), logging.INFO)
logging.basicConfig(filename='bot.log', level=log_level,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# **10. بيانات التدريب (موسعة)**
error_data = [] # تهيئة قائمة بيانات التدريب

# **11. تحويل البيانات إلى تنسيق مناسب للنموذج**
def prepare_data(data):
    features = []
    labels_error = []
    labels_proxy = []
    for item in data:
        feature = [
            1 if item["user_agent"] in user_agents else 0,
            1 if item["proxy_used"] else 0,
            item["status_code"],
            item["response_time"],
            item["load_time"],
            item["page_size"],
            item["num_requests"],
            1 if item["page_type"] == "product" else 0,
            1 if item["page_type"] == "blog" else 0,
            1 if item["page_type"] == "login" else 0,
        ]
        label_error = 1 if item["error_type"] != "success" else 0
        label_proxy = 1 if item["proxy_valid"] else 0
        features.append(feature)
        labels_error.append(label_error)
        labels_proxy.append(label_proxy)
    return torch.tensor(features, dtype=torch.float32), torch.tensor(labels_error, dtype=torch.float32), torch.tensor(labels_proxy, dtype=torch.float32)

# **12. تدريب نموذج التعلم متعدد المهام**
def train_model(model, optimizer, features, labels_error, labels_proxy, epochs=100):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss_error = nn.BCELoss()(outputs[0].squeeze(), labels_error)
        loss_proxy = nn.BCELoss()(outputs[1].squeeze(), labels_proxy)
        loss = loss_error + loss_proxy
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# **13. دالة للتحقق من صحة الرابط**
def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

# **14. دالة لتدوير الوكلاء**
def rotate_proxy():
    global current_proxy_index
    with lock:
        current_proxy_index = (current_proxy_index + 1) % len(proxy_list)
        return proxy_list[current_proxy_index]

# **15. دالة للتحقق من صحة الوكيل (باستخدام النموذج)**
def validate_proxy(proxy, user_agent, status_code, response_time, load_time, page_size, num_requests, page_type):
    model.eval()
    feature = torch.tensor([1 if user_agent in user_agents else 0, 1, status_code, response_time, load_time, page_size, num_requests, 1 if page_type == "product" else 0, 1 if page_type == "blog" else 0, 1 if page_type == "login" else 0], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(feature)
        proxy_valid_prob = outputs[1].item()
    return proxy_valid_prob > CONFIG["PROXY_VALIDATION_THRESHOLD"]

# **16. دالة للحصول على الموقع الجغرافي من عنوان IP**
def get_location_from_ip(ip_address):
    try:
        geolocator = Nominatim(user_agent="visit_bot")
        location = geolocator.geocode(ip_address)
        if location:
            return location.latitude, location.longitude
        else:
            return None
    except:
        return None

# **17. دالة لمحاكاة الزيارة (مع تحسينات الذكاء الاصطناعي)**
def simulate_visit(retry_count=0):
    global visits_count, successful_visits, failed_visits

    # تدوير الوكيل بشكل احتمالي
    if proxy_rotation_enabled and random.random() < proxy_rotation_frequency:
        current_proxy = next(proxy_cycle)
    else:
        current_proxy = PROXY_URL

    proxies = {
        'http': current_proxy,
        'https': current_proxy,
    }

    # تدوير وكيل المستخدم
    current_user_agent = next(user_agent_cycle)
    headers = {
        'User-Agent': current_user_agent,
    }

    start_time = time.time()
    try:
        time.sleep(random.uniform(0.5, 2.5))
        response = requests.get(target_url, proxies=proxies, headers=headers, timeout=10)
        end_time = time.time()
        response_time = end_time - start_time

        # تحليل محتوى الصفحة
        soup = BeautifulSoup(response.content, 'html.parser')
        page_size = len(response.content)
        num_requests = len(soup.find_all())  # تقدير عدد الطلبات بناءً على عدد العناصر

        with lock:
            visits_count += 1
            if response.status_code == 200:
                successful_visits += 1
                logging.info(f"Successful visit to {target_url} with proxy {current_proxy}, response time: {response_time:.2f}s")
                # جمع البيانات للتدريب
                if data_collection_enabled:
                    page_type = "unknown"  # يمكنك محاولة تحديد نوع الصفحة هنا
                    proxy_valid = True
                    new_data = {"error_type": "success", "user_agent": current_user_agent, "proxy_used": True, "status_code": response.status_code, "response_time": response_time, "proxy_valid": proxy_valid, "page_type": page_type, "load_time": response_time * 1000, "page_size": page_size, "num_requests": num_requests}
                    error_data.append(new_data)
            else:
                failed_visits += 1
                log_error(f"HTTP error: {response.status_code} for URL: {target_url} with proxy {current_proxy}, response time: {response_time:.2f}s")
                # جمع البيانات للتدريب
                if data_collection_enabled:
                    page_type = "unknown"  # يمكنك محاولة تحديد نوع الصفحة هنا
                    proxy_valid = False
                    new_data = {"error_type": "http_error", "user_agent": current_user_agent, "proxy_used": True, "status_code": response.status_code, "response_time": response_time, "proxy_valid": proxy_valid, "page_type": page_type, "load_time": response_time * 1000, "page_size": page_size, "num_requests": num_requests}
                    error_data.append(new_data)

                if ai_enabled and smart_error_handling:
                    # استخدام النموذج للتنبؤ بالخطأ
                    features = torch.tensor([1 if headers['User-Agent'] in user_agents else 0, 1, response.status_code, response_time, 0, 0, 0, 0, 0, 0], dtype=torch.float32).unsqueeze(0)
                    model.eval()
                    with torch.no_grad():
                        outputs = model(features)
                        error_prob = outputs[0].item()
                    if error_prob > 0.5:  # عتبة الخطأ
                        print("Error predicted by model, adjusting parameters...")
                        if ai_adjust_concurrency:
                            adjust_concurrency()
    except requests.exceptions.ProxyError as e:
        end_time = time.time()
        response_time = end_time - start_time
        with lock:
            failed_visits += 1
        log_error(f"Proxy error: {e} for URL: {target_url} with proxy {current_proxy}, response time: {response_time:.2f}s")
        # جمع البيانات للتدريب
        if data_collection_enabled:
            page_type = "unknown"  # يمكنك محاولة تحديد نوع الصفحة هنا
            proxy_valid = False
            new_data = {"error_type": "proxy_error", "user_agent": current_user_agent, "proxy_used": True, "status_code": 0, "response_time": response_time, "proxy_valid": proxy_valid, "page_type": page_type, "load_time": response_time * 1000, "page_size": 0, "num_requests": 0}
            error_data.append(new_data)
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        response_time = end_time - start_time
        with lock:
            failed_visits += 1
        log_error(f"Request Exception: {e} for URL: {target_url} with proxy {current_proxy}, response time: {response_time:.2f}s")
        # جمع البيانات للتدريب
        if data_collection_enabled:
            page_type = "unknown"  # يمكنك محاولة تحديد نوع الصفحة هنا
            proxy_valid = False
            new_data = {"error_type": "request_exception", "user_agent": current_user_agent, "proxy_used": True, "status_code": 0, "response_time": response_time, "proxy_valid": proxy_valid, "page_type": page_type, "load_time": response_time * 1000, "page_size": 0, "num_requests": 0}
            error_data.append(new_data)

# **18. دالة لتسجيل الأخطاء**
def log_error(message):
    logging.error(message)
    print(f"Error: {message}")

# **19. دالة لتشغيل الزيارات مع التحكم في التزامن**
def run_visits(concurrency, visits_per_day, stop_event):
    global running
    running = True
    visits_per_minute = visits_per_day / (24 * 60)
    interval = 60 / visits_per_minute
    while running and not stop_event.is_set():
        threads = []
        for _ in range(concurrency):
            t = threading.Thread(target=simulate_visit)
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        time.sleep(interval)

# **20. دالة لتعديل التزامن (بناءً على الذكاء الاصطناعي)**
def adjust_concurrency():
    global current_concurrency, current_visits_per_day
    cpu_usage = psutil.cpu_percent(interval=1)
    mem_usage = psutil.virtual_memory().percent
    print(f"CPU Usage: {cpu_usage}%, Memory Usage: {mem_usage}%")
    # يمكن إضافة المزيد من المنطق هنا بناءً على مقاييس الأداء وأنواع الأخطاء
    if cpu_usage > 80 or mem_usage > 80:
        current_concurrency = max(CONFIG["MIN_CONCURRENCY"], int(current_concurrency * (1 - concurrency_adjustment_factor)))
        print(f"Reducing concurrency to {current_concurrency} due to high resource usage.")
    else:
        if current_concurrency < CONFIG["DEFAULT_CONCURRENCY"]:
            current_concurrency = min(CONFIG["DEFAULT_CONCURRENCY"], int(current_concurrency * (1 + concurrency_adjustment_factor)))
            print(f"Increasing concurrency to {current_concurrency}.")
        elif current_concurrency < CONFIG["MAX_CONCURRENCY"]:
            current_concurrency = min(CONFIG["MAX_CONCURRENCY"], int(current_concurrency * (1 + concurrency_adjustment_factor)))
            print(f"Increasing concurrency to {current_concurrency}.")
    return current_concurrency

# **21. دالة لتعديل عدد الزيارات اليومية (بناءً على الذكاء الاصطناعي)**
def adjust_visits_per_day():
    global current_visits_per_day
    # يمكن إضافة المزيد من المنطق هنا بناءً على مقاييس الأداء وأنواع الأخطاء
    # على سبيل المثال، إذا كان معدل النجاح مرتفعًا، يمكن زيادة عدد الزيارات
    # وإذا كان معدل الفشل مرتفعًا، يمكن تقليل عدد الزيارات
    if successful_visits > visits_count * success_rate_threshold:
        current_visits_per_day = min(int(current_visits_per_day * (1 + visits_adjustment_factor)), 15000)  # زيادة بنسبة 5%
        print(f"Increasing visits per day to {current_visits_per_day}.")
    elif failed_visits > visits_count * failure_rate_threshold:
        current_visits_per_day = max(int(current_visits_per_day * (1 - visits_adjustment_factor)), 2000)  # تقليل بنسبة 5%
        print(f"Reducing visits per day to {current_visits_per_day}.")
    return current_visits_per_day

# **22. دالة لمراقبة الأداء وتعديل الخوارزميات**
def monitor_performance():
    global successful_visits, failed_visits, visits_count, current_concurrency, current_visits_per_day

    # حساب المقاييس
    success_rate = successful_visits / visits_count if visits_count > 0 else 0
    failure_rate = failed_visits / visits_count if visits_count > 0 else 0
    # يمكن إضافة المزيد من المقاييس هنا مثل وقت الاستجابة ومعدل الارتداد

    print(f"Success Rate: {success_rate:.2f}, Failure Rate: {failure_rate:.2f}")

    # تعديل الخوارزميات
    if success_rate < success_rate_threshold:
        print("Low success rate, adjusting parameters...")
        # يمكن إضافة المزيد من المنطق هنا لتعديل الخوارزميات
        # على سبيل المثال، يمكن زيادة تدوير الوكلاء أو تغيير User-Agent
        if proxy_rotation_enabled:
            print("Increasing proxy rotation frequency...")
            global proxy_rotation_frequency
            proxy_rotation_frequency = min(proxy_rotation_frequency + 0.1, 1.0)  # زيادة بنسبة 10%
        else:
            print("Enabling proxy rotation...")
            global proxy_rotation_enabled
            proxy_rotation_enabled = True

    if failure_rate > failure_rate_threshold:
        print("High failure rate, adjusting parameters...")
        # يمكن إضافة المزيد من المنطق هنا لتعديل الخوارزميات
        # على سبيل المثال، يمكن تقليل التزامن أو تغيير الوكلاء
        current_concurrency = max(CONFIG["MIN_CONCURRENCY"], int(current_concurrency * (1 - concurrency_adjustment_factor)))
        print(f"Reducing concurrency to {current_concurrency}.")

    # إعادة تعيين العدادات
    with lock:
        successful_visits = 0
        failed_visits = 0
        visits_count = 0

# **23. دالة لجمع البيانات تلقائيًا**
def collect_data():
    global error_data
    print("Collecting data...")
    # يمكنك جمع البيانات من سجلات البوت هنا
    # على سبيل المثال، يمكنك قراءة البيانات من ملف السجل
    # أو يمكنك جمع البيانات من خلال مراقبة سلوك البوت
    # يجب التأكد من أن البيانات المجمعة دقيقة وكاملة وتمثل مجموعة واسعة من السيناريوهات
    # يجب أيضًا التأكد من أن البيانات المجمعة متوازنة
    # يمكن استخدام تقنيات مثل إعادة التشكيل (resampling) أو ترجيح الفئات (class weighting)
    # إذا كانت البيانات غير متوازنة
    print("Data collection complete.")

# **24. دالة لتحسين نموذج التعلم متعدد المهام**
def optimize_model():
    global model, optimizer, features, labels_error, labels_proxy
    print("Optimizing model...")

    # 1. إعداد البيانات
    features, labels_error, labels_proxy = prepare_data(error_data)

    # 2. تحديد شبكة المعلمات
    param_grid = {
        'hidden_size': [16, 32, 64],
        'learning_rate': [0.001, 0.01, 0.1],
        'epochs': [50, 100, 150]
    }

    # 3. تعريف دالة الهدف
    def objective(trial):
        hidden_size = trial.suggest_categorical('hidden_size', [16, 32, 64])
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        epochs = trial.suggest_int('epochs', 50, 150)

        # تهيئة النموذج
        model = MultiTaskModel(input_size, hidden_size, num_tasks)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # تدريب النموذج
        train_model(model, optimizer, features, labels_error, labels_proxy, epochs)

        # تقييم النموذج
        model.eval()
        with torch.no_grad():
            outputs = model(features)
            loss_error = nn.BCELoss()(outputs[0].squeeze(), labels_error)
            loss_proxy = nn.BCELoss()(outputs[1].squeeze(), labels_proxy)
            loss = loss_error + loss_proxy

        return loss.item()

    # 4. تشغيل التحسين
    import optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=num_trials)

    # 5. الحصول على أفضل المعلمات
    best_params = study.best_params
    print(f"Best parameters: {best_params}")

    # 6. إعادة تهيئة النموذج بأفضل المعلمات
    model = MultiTaskModel(input_size, best_params['hidden_size'], num_tasks)
    optimizer = optim.Adam(model.parameters(), lr=best_params['learning_rate'])

    # 7. إعادة تدريب النموذج بأفضل المعلمات
    train_model(model, optimizer, features, labels_error, labels_proxy, best_params['epochs'])

    print("Model optimization complete.")

# **25. دالة لمراقبة موارد النظام**
def monitor_resources(stop_event):
    while running and not stop_event.is_set():
        cpu_usage = psutil.cpu_percent(interval=1)
        mem_usage = psutil.virtual_memory().percent
        print(f"استخدام المعالج: {cpu_usage}%, استخدام الذاكرة: {mem_usage}%")
        if cpu_usage > 80 or mem_usage > 80:
            adjust_concurrency()
        if ai_enabled and ai_adjust_visits:
            adjust_visits_per_day()
        time.sleep(60)

# **26. دالة رئيسية للمراقبة الدورية**
def ai_monitor(stop_event):
    while running and not stop_event.is_set():
        time.sleep(ai_monitor_interval)
        monitor_performance()

# **27. دالة لجمع البيانات بشكل دوري**
def data_collection_loop(stop_event):
    while running and not stop_event.is_set():
        time.sleep(data_collection_interval)
        if data_collection_enabled:
            collect_data()
            if optimize_model_enabled:
                optimize_model() # تحسين النموذج بعد جمع البيانات

# **28. معالجات الأوامر (محدثة إلى async)**
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global running, bot_running
    with lock:
        if not bot_running:
            bot_running = True
            await update.message.reply_text('تم تشغيل البوت بنجاح ولكن لم يبدأ أي زيارات بعد.')
        else:
            await update.message.reply_text('البوت يعمل بالفعل.')

async def startvisits(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global running, target_url, bot_running, stop_event
    if context.args:
        target_url = context.args[0]
        if not is_valid_url(target_url):
            await update.message.reply_text('الرابط غير صالح. الرجاء إدخال رابط صحيح.')
            return
        with lock:
            if not bot_running:
                bot_running = True
                running = True
                concurrency = adjust_concurrency()
                stop_event = threading.Event()
                threading.Thread(target=run_visits, args=(concurrency, current_visits_per_day, stop_event), daemon=True).start()
                threading.Thread(target=monitor_resources, args=(stop_event,), daemon=True).start()
                # بدء مراقبة الذكاء الاصطناعي
                threading.Thread(target=ai_monitor, args=(stop_event,), daemon=True).start()
                # بدء جمع البيانات بشكل دوري
                threading.Thread(target=data_collection_loop, args=(stop_event,), daemon=True).start()
                await update.message.reply_text(f'بدأت الزيارات بنجاح إلى: {target_url}')
            else:
                if not running:
                    running = True
                    concurrency = adjust_concurrency()
                    stop_event = threading.Event()
                    threading.Thread(target=run_visits, args=(concurrency, current_visits_per_day, stop_event), daemon=True).start()
                    threading.Thread(target=monitor_resources, args=(stop_event,), daemon=True).start()
                    # بدء مراقبة الذكاء الاصطناعي
                    threading.Thread(target=ai_monitor, args=(stop_event,), daemon=True).start()
                    # بدء جمع البيانات بشكل دوري
                    threading.Thread(target=data_collection_loop, args=(stop_event,), daemon=True).start()
                    await update.message.reply_text(f'بدأت الزيارات بنجاح إلى: {target_url}')
                else:
                    await update.message.reply_text(f'البوت يقوم بزيارة بالفعل: {target_url}. تم تحديث الرابط إلى: {target_url}')
    else:
        await update.message.reply_text('الرجاء تقديم رابط. الاستخدام: /startvisits <url>')

async def stop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global running, bot_running, stop_event
    with lock:
        running = False
        bot_running = False
        if stop_event:
            stop_event.set()
    await update.message.reply_text('تم إيقاف البوت والزيارات بنجاح.')

async def stats(update: Update, context: ContextTypes.DEFAULT_TYPE):
    with lock:
        await update.message.reply_text(f'إجمالي الزيارات: {visits_count}\nالزيارات الناجحة: {successful_visits}\nالزيارات الفاشلة: {failed_visits}')

async def errors(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text('تقارير الأخطاء غير متاحة بعد.')

async def setconcurrency(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_concurrency
    if context.args:
        try:
            concurrency = int(context.args[0])
            if CONFIG["MIN_CONCURRENCY"] <= concurrency <= CONFIG["MAX_CONCURRENCY"]:
                with lock:
                    current_concurrency = concurrency
                await update.message.reply_text(f'تم تعيين التزامن إلى: {concurrency}')
            else:
                await update.message.reply_text(f'يجب أن يكون التزامن بين {CONFIG["MIN_CONCURRENCY"]} و {CONFIG["MAX_CONCURRENCY"]}.')
        except ValueError:
            await update.message.reply_text('الرجاء تقديم رقم صحيح للتزامن.')
    else:
        await update.message.reply_text('الرجاء تقديم قيمة للتزامن. الاستخدام: /setconcurrency <number>')

async def settarget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global current_visits_per_day
    if context.args:
        try:
            target = int(context.args[0])
            if target > 0:
                with lock:
                    current_visits_per_day = target
                await update.message.reply_text(f'تم تعيين هدف الزيارات اليومي إلى: {target}')
            else:
                await update.message.reply_text('يجب أن يكون الهدف رقمًا موجبًا.')
        except ValueError:
            await update.message.reply_text('الرجاء تقديم رقم صحيح للهدف.')
    else:
        await update.message.reply_text('الرجاء تقديم قيمة للهدف. الاستخدام: /settarget <number>')

async def sysinfo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    cpu_freq = psutil.cpu_freq()
    mem = psutil.virtual_memory()
    swap = psutil.swap_memory()
    disk = psutil.disk_usage('/')
    boot_time = time.ctime(psutil.boot_time())

    response = (
        f"🖥 معلومات النظام:\n"
        f"------------------
        #pip install requests psutil numpy scikit-learn python-telegram-bot geopy beautifulsoup4 torch optuna
