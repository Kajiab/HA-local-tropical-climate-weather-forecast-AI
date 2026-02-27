#transform CSV file format to Nixtla

import pandas as pd
import numpy as np

def prepare_for_nixtla(file_path):
    # 1. โหลดข้อมูลจาก CSV (ใส่ชื่อ Column ให้ตรงกับที่เราออกแบบใน Automation)
    columns = [
        'timestamp', 'temp', 'hum', 'p0', 'dew', 
        'p_change', 'h_change', 't_change', 'd_change',
        'ecmwf_press', 'ecmwf_rain', 'home_htd, 'actual'
    ]
    
    df = pd.read_csv(file_path, names=columns)
    
    # 2. แปลง Timestamp เป็น Datetime object
    df['ds'] = pd.to_datetime(df['timestamp'])
    
    # 3. สร้าง Label 'y' (เป้าหมายที่จะทำนาย)
    # แปลง "RAIN_ACTUAL" เป็น 1 และที่เหลือเป็น 0 (สำหรับ Classification)
    # หรือถ้าคุณเจี๊ยบจะทำนายปริมาณน้ำฝน ก็ใช้ค่า ecmwf_rain หรือค่าจาก rain gauge
    df['y'] = df['actual'].apply(lambda x: 1 if x == 'RAIN_ACTUAL' else 0)
    
    # 4. ระบุ Unique ID (สำคัญมากสำหรับ Nixtla)
    df['unique_id'] = 'bangkok_station_01'
    
    # 5. จัดการ Exogenous Variables (ตัวแปรเสริมที่จะช่วยให้ AI ฉลาดขึ้น)
    # เราเลือกเอาค่า Change ต่างๆ และค่าจาก ECMWF มาเป็นฟีเจอร์ช่วยทำนาย
    exog_cols = ['temp', 'hum', 'p0', 'p_change', 'h_change', 'd_change', 'ecmwf_press', 'ecmwf_rain']
    
    # 6. คัดแยกเฉพาะ Column ที่ Nixtla ต้องใช้
    nixtla_df = df[['unique_id', 'ds', 'y'] + exog_cols].copy()
    
    # เรียงลำดับเวลา (เผื่อคุณเจี๊ยบกดบันทึกข้ามไปข้ามมา)
    nixtla_df = nixtla_df.sort_values(['unique_id', 'ds']).reset_index(drop=True)
    
    return nixtla_df

# การใช้งาน
# df_ready = prepare_for_nixtla('weather_dataset.csv')
# print(df_ready.head())
