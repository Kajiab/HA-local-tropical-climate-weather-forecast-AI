#TFT transfer learning

from neuralforecast import NeuralForecast
from neuralforecast.models import TFT
from neuralforecast.losses.pytorch import DistributionLoss

# 1. เตรียมข้อมูลจาก Script เดิมของคุณเจี๊ยบ
df_ready = prepare_for_nixtla('weather_dataset.csv')

# 2. กำหนดรายชื่อตัวแปรเสริม (Exogenous Variables)
# ข้อมูลอดีตที่เรามีจากเซนเซอร์
hist_exog = ['temp', 'hum', 'p0', 'dew', 'p_change', 't_change',  'h_change', 'd_change']
# ข้อมูลที่เรารู้อนาคตจาก ECMWF
futr_exog = ['ecmwf_press', 'ecmwf_rain']

# 3. นิยามโมเดล TFT
# h: ระยะเวลาที่ต้องการพยากรณ์ล่วงหน้า (หน่วยตามความถี่ข้อมูล เช่น 6 ชั่วโมง)
# input_size: จำนวนข้อมูลย้อนหลังที่จะให้ AI ดู (เช่น ดูย้อนหลัง 24 ชั่วโมง)
models = [
    TFT(h=6, 
        input_size=24,
        hist_exog_list=hist_exog,
        futr_exog_list=futr_exog,
        scaler_type='robust', # ช่วยจัดการ Outlier ของเซนเซอร์ได้ดี
        max_steps=500,        # จำนวนรอบในการเทรน (ปรับเพิ่มได้ถ้าข้อมูลเยอะ)
        learning_rate=0.001)
]

# 4. สร้างระบบ NeuralForecast
# freq: ความถี่ของข้อมูล ('H' คือรายชั่วโมง)
nf = NeuralForecast(models=models, freq='H')

# 5. เริ่มต้นการเทรน (Fit)
print("Starting Training...")
nf.fit(df=df_ready)

# 6. บันทึก Model ไว้ใช้งาน (ไม่ต้องเทรนใหม่ทุกครั้ง)
nf.save(path='./weather_model/')
print("Model saved successfully!")

# 7. ทดลองพยากรณ์ (Forecast)
# ต้องป้อนข้อมูลอนาคตของ futr_exog (ECMWF) เข้าไปด้วย
# forecast_df = nf.predict(futr_df=df_with_future_ecmwf)
