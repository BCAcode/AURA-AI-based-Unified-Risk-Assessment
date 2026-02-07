#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import Circle
import matplotlib as mpl
import json
import os
import ssl
from io import BytesIO
from scipy.ndimage import gaussian_filter, label, median_filter
from PIL import Image
from datetime import datetime
import time
from pathlib import Path
import sys

# SSL Bypass
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    print("FATAL ERROR: 'requests' library not found!")
    print("Install with: pip install requests urllib3 --break-system-packages")
    sys.exit(1)

try:
    import mlx.core as mx
    import mlx.nn as nn
    METAL_AVAILABLE = True
except ImportError:
    METAL_AVAILABLE = False
    print("FATAL ERROR: MLX not found!")
    print("This system requires Apple Silicon MLX framework.")
    print("Install with: pip install mlx mlx-nn --break-system-packages")
    sys.exit(1)

mpl.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')

HEIGHT = 144
WIDTH = int(HEIGHT * 16 / 9)

# AUTO MODEL DETECTION
def find_model_file():
    model_name = "AURA_Ignition_5_Model.npz"
    
    search_paths = [
        Path.cwd() / model_name,
        Path.cwd() / "models" / model_name,
        Path.cwd().parent / "models" / model_name,
        Path.home() / model_name,
        Path.home() / "Desktop" / model_name,
        Path.home() / "Desktop" / "ml" / model_name,
        Path.home() / "Documents" / model_name,
        Path.home() / "Downloads" / model_name,
    ]
    
    print(f"[MODEL] Searching for {model_name}...")
    
    for path in search_paths:
        if path.exists():
            print(f"[OK] Model found: {path}")
            return str(path)
            
    print(f"[SEARCH] Model not found in default paths. Attempting system-wide search...")
    
    # Try mdfind (macOS Spotlight) - Finds file anywhere instantly
    try:
        import subprocess
        print("[SEARCH] Querying Spotlight database...")
        cmd = ['mdfind', '-name', model_name]
        result = subprocess.run(cmd, capture_output=True, text=True)
        found_paths = [p.strip() for p in result.stdout.split('\n') if p.strip()]
        
        # Filter and pick the first valid one
        for p in found_paths:
            if p.endswith(model_name) and os.path.exists(p):
                print(f"[OK] Model found via Spotlight: {p}")
                return p
    except Exception as e:
        print(f"[WARN] Spotlight search failed: {e}")
    
    print(f"\n{'='*70}")
    print(f"FATAL ERROR: Model file '{model_name}' not found!")
    print(f"{'='*70}")
    print(f"Searched locations:")
    for path in search_paths:
        print(f"  - {path}")
    print(f"  - [System Wide Search]")
    print(f"\nSOLUTION: Place '{model_name}' in your Documents or Desktop folder.")
    print(f"{'='*70}\n")
    return None

MODEL_PATH = find_model_file()

if MODEL_PATH is None:
    print("\nSYSTEM CANNOT START WITHOUT MODEL FILE")
    print("Exiting...")
    sys.exit(1)

STATS_PATH = None

MODEL_INFO = {
    'name': 'AURA Ignition-5',
    'version': '2.0',
    'parameters': 563969,
    'architecture': '6-layer DNN',
    'layers': '20→256→512→512→256→128→1',
    'type': 'Deep Learning Regressor'
}

class AppState:
    def __init__(self):
        self.running = True
        self.current_tree_idx = 0
        self.current_season_idx = 1
        self.current_bg_mode = 2 
        self.drought_factor = 0.5
        self.days_since_rain = 0
        self.current_lat = 39.7536
        self.current_lon = 27.4912
        self.view_height = 0.015
        self.current_date = datetime.now()
        self.weather_data_loaded = False
        self.textbox_active = False
        self.auto_mode = True
        self.propagation_enabled = False

state = AppState()
fire_patches = np.zeros((HEIGHT, WIDTH))
water_mask = np.zeros((HEIGHT, WIDTH), dtype=bool)
propagation_map = np.zeros((HEIGHT, WIDTH))

tree_types = {
    "Cam": {"moisture": 0.15, "density": 0.7, "color": "#228B22"},
    "Mese": {"moisture": 0.25, "density": 0.9, "color": "#006400"},
    "Cinar": {"moisture": 0.10, "density": 0.5, "color": "#00ff7f"},
    "Kayin": {"moisture": 0.20, "density": 0.8, "color": "#2e8b57"},
    "Kavak": {"moisture": 0.08, "density": 0.4, "color": "#7fff00"},
    "Maki": {"moisture": 0.05, "density": 0.5, "color": "#32cd32"},
    "Cali": {"moisture": 0.03, "density": 0.3, "color": "#adff2f"},
}

seasons = {
    "Ilkbahar": {"temp_mod": 0.0, "humidity_mod": 0.1, "risk_mod": 0.7},
    "Yaz": {"temp_mod": 0.3, "humidity_mod": -0.2, "risk_mod": 1.0},
    "Sonbahar": {"temp_mod": -0.1, "humidity_mod": 0.0, "risk_mod": 0.8},
    "Kis": {"temp_mod": -0.3, "humidity_mod": 0.2, "risk_mod": 0.3}
}

def get_season_from_date(date):
    month = date.month
    if month in [3, 4, 5]:
        return "Ilkbahar"
    elif month in [6, 7, 8]:
        return "Yaz"
    elif month in [9, 10, 11]:
        return "Sonbahar"
    else:
        return "Kis"

def fractal_dimension(binary_mask, threshold=0.5):
    Z = (binary_mask > threshold).astype(int)
    if Z.sum() == 0:
        return 1.0
    
    min_dim = min(Z.shape)
    if min_dim < 4:
        return 1.0
    
    max_exp = int(np.log2(min_dim))
    sizes = 2**np.arange(1, min(max_exp, 6), 1)
    
    if len(sizes) < 2:
        return 1.0
    
    counts = []
    for size in sizes:
        try:
            n_boxes_y = Z.shape[0] // size
            n_boxes_x = Z.shape[1] // size
            
            if n_boxes_y == 0 or n_boxes_x == 0:
                continue
                
            boxes = 0
            for i in range(n_boxes_y):
                for j in range(n_boxes_x):
                    box = Z[i*size:(i+1)*size, j*size:(j+1)*size]
                    if np.any(box):
                        boxes += 1
            
            if boxes > 0:
                counts.append(boxes)
        except:
            continue
    
    if len(counts) < 2:
        return 1.0
    
    valid_sizes = sizes[:len(counts)]
    try:
        coeffs = np.polyfit(np.log(valid_sizes), np.log(counts), 1)
        fd = -coeffs[0]
        fd = np.clip(fd, 1.0, 2.0)
        return fd
    except:
        return 1.0

def calculate_real_area(pixel_count, lat, view_height):
    lat_rad = np.radians(lat)
    dlat = view_height * (HEIGHT / WIDTH)
    dlon = view_height / np.cos(lat_rad)
    pixel_height_km = dlat / HEIGHT
    pixel_width_km = dlon / WIDTH
    pixel_area_km2 = pixel_height_km * pixel_width_km
    total_area_km2 = pixel_count * pixel_area_km2
    total_area_m2 = total_area_km2 * 1_000_000
    return total_area_m2

def calculate_propagation_potential(risk_map, wind_angle, wind_strength, slope_map, aspect_map, veg_density_map, tree_moisture):
    propagation = risk_map.copy()
    wind_rad = np.radians(wind_angle)
    wind_dx = np.cos(wind_rad)
    wind_dy = np.sin(wind_rad)
    
    gy, gx = np.gradient(propagation)
    wind_boost = np.zeros_like(propagation)
    
    for i in range(1, HEIGHT-1):
        for j in range(1, WIDTH-1):
            if propagation[i, j] > 0.3:
                target_i = int(i + wind_dy * 3)
                target_j = int(j + wind_dx * 3)
                
                if 0 <= target_i < HEIGHT and 0 <= target_j < WIDTH:
                    wind_boost[target_i, target_j] += propagation[i, j] * wind_strength * 0.3
    
    propagation = propagation + wind_boost
    slope_effect = 1.0 + slope_map * 0.3
    propagation *= slope_effect
    veg_effect = 0.5 + veg_density_map * 0.5
    propagation *= veg_effect
    moisture_effect = 1.0 - tree_moisture * 0.5
    propagation *= moisture_effect
    propagation = gaussian_filter(propagation, sigma=2.0)
    propagation = np.clip(propagation, 0, 1)
    propagation[propagation < 0.2] = 0
    
    return propagation

def format_area(area_m2):
    if area_m2 < 10000:
        return f"{area_m2:.0f} m²"
    elif area_m2 < 1_000_000:
        hectares = area_m2 / 10000
        return f"{hectares:.2f} ha"
    else:
        km2 = area_m2 / 1_000_000
        return f"{km2:.3f} km²"

def generate_terrain():
    terrain = np.random.randn(HEIGHT, WIDTH)
    terrain = gaussian_filter(terrain, sigma=8)
    terrain = (terrain - terrain.min()) / (terrain.max() - terrain.min())
    
    elevation = terrain * 1000
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gx**2 + gy**2)
    slope = np.clip(slope / (slope.max() + 1e-6), 0, 1)
    aspect = np.arctan2(gy, gx)
    south_facing = np.cos(aspect)
    
    return elevation, slope, south_facing

def generate_drought_map():
    drought = np.random.randn(HEIGHT, WIDTH)
    drought = gaussian_filter(drought, sigma=15)
    drought = (drought - drought.min()) / (drought.max() - drought.min())
    return drought

def generate_vegetation_density():
    veg = np.random.randn(HEIGHT, WIDTH)
    veg = gaussian_filter(veg, sigma=10)
    veg = (veg - veg.min()) / (veg.max() - veg.min())
    return veg

def encode_wind_angle(angle):
    rad = np.radians(angle)
    return np.sin(rad), np.cos(rad)

def calculate_solar_radiation(hour, season_name):
    radiation = np.cos(np.pi * (hour - 12) / 12)
    radiation = max(0, radiation)
    
    season_factor = {
        "Yaz": 1.0,
        "Ilkbahar": 0.8,
        "Sonbahar": 0.6,
        "Kis": 0.4
    }
    
    return radiation * season_factor.get(season_name, 0.8)

def fetch_weather_data(lat, lon, target_date):
    try:
        today = datetime.now().date()
        target_date_obj = target_date.date() if isinstance(target_date, datetime) else target_date
        
        if target_date_obj <= today:
            days_diff = (today - target_date_obj).days
            if days_diff > 365:
                return None
            url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={target_date_obj}&end_date={target_date_obj}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant&timezone=auto"
        else:
            days_diff = (target_date_obj - today).days
            if days_diff > 16:
                return None
            url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max,winddirection_10m_dominant&timezone=auto&forecast_days=16"
        
        print(f"[WEATHER] Fetching: {target_date_obj}")
        res = requests.get(url, timeout=15, verify=False)
        
        if res.status_code == 200:
            data = res.json()
            daily = data.get('daily', {})
            dates = daily.get('time', [])
            target_str = str(target_date_obj)
            
            if target_str in dates:
                idx = dates.index(target_str)
                temp_max = daily.get('temperature_2m_max', [None])[idx]
                temp_min = daily.get('temperature_2m_min', [None])[idx]
                precipitation = daily.get('precipitation_sum', [None])[idx]
                wind_speed = daily.get('windspeed_10m_max', [None])[idx]
                wind_dir = daily.get('winddirection_10m_dominant', [None])[idx]
                
                weather = {
                    'temperature': (temp_max + temp_min) / 2 if temp_max and temp_min else 25.0,
                    'precipitation': precipitation if precipitation else 0.0,
                    'wind_speed': wind_speed if wind_speed else 15.0,
                    'wind_direction': wind_dir if wind_dir else 90.0,
                    'humidity': 0.3 if precipitation and precipitation > 1.0 else 0.2
                }
                
                print(f"[OK] Weather loaded: {weather['temperature']:.1f}C")
                return weather
        return None
    except Exception as e:
        print(f"Weather error: {e}")
        return None

def fetch_current_time(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m&timezone=auto"
        res = requests.get(url, timeout=10, verify=False)
        
        if res.status_code == 200:
            data = res.json()
            current_time_str = data.get('current', {}).get('time', '')
            if current_time_str:
                hour = int(current_time_str.split('T')[1].split(':')[0])
                print(f"[OK] Local time: {hour:02d}:00")
                return hour
    except Exception as e:
        print(f"[WARN] Time fetch error: {e}")
    
    from datetime import datetime
    utc_now = datetime.utcnow()
    offset_hours = round(lon / 15.0)
    local_hour = (utc_now.hour + offset_hours) % 24
    print(f"[OK] Estimated time: {local_hour:02d}:00")
    return local_hour

def fetch_map_data(lat, lon):
    global elevation_map, slope_map, aspect_map, satellite_img, veg_density_map, drought_map, water_mask
    
    ui_ready = 'im_bg_satellite' in globals() and im_bg_satellite is not None
    
    try:
        lat_rad = np.radians(lat)
        view_scale = state.view_height
        dlat = view_scale * (HEIGHT / WIDTH)
        dlon = view_scale / np.cos(lat_rad)
        west, south, east, north = lon-dlon/2, lat-dlat/2, lon+dlon/2, lat+dlat/2
        
        print(f"[SAT] Fetching satellite imagery...")
        sat_url = f"https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export?bbox={west},{south},{east},{north}&bboxSR=4326&imageSR=4326&size={WIDTH},{HEIGHT}&format=png&transparent=false&f=image"
        
        sat_success = False
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        }

        for attempt in range(2):
            try:
                res = requests.get(sat_url, timeout=15, verify=False, headers=headers)
                if res.status_code == 200 and len(res.content) > 1000:
                    try:
                        img = Image.open(BytesIO(res.content))
                        if img.mode != 'RGB':
                            img = img.convert('RGB')
                        img_sat = img.resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
                        satellite_img = np.array(img_sat, dtype=np.float32) / 255.0
                        
                        if satellite_img.std() > 0.01:
                            print(f"[OK] Satellite loaded")
                            sat_success = True
                            break
                    except Exception as e:
                        print(f"[WARN] Image processing error: {e}")
                else:
                    print(f"[WARN] Server error or empty content")
            except Exception as e:
                print(f"[WARN] Satellite fetch failed: {str(e)[:80]}")
            
            if attempt < 1:
                time.sleep(0.5)
        
        if not sat_success:
            print(f"[INFO] Using synthetic satellite base")
            base = np.random.rand(HEIGHT, WIDTH, 3) * 0.2
            base[:,:,1] += 0.3
            satellite_img = np.clip(base, 0, 1).astype(np.float32)
        
        if ui_ready:
            im_bg_satellite.set_data(satellite_img)
            im_p1_sat.set_data(satellite_img)
            im_p2_sat.set_data(satellite_img)
        
        r, g, b = satellite_img[:,:,0], satellite_img[:,:,1], satellite_img[:,:,2]
        water_mask = (b > g) & (b > r) & (r < 0.25) & (b > 0.2)
        veg_proxy = (2*g - r - b) / (g + r + b + 1e-6)
        veg_density_map = np.clip((veg_proxy - veg_proxy.min()) / (veg_proxy.max() - veg_proxy.min() + 1e-6), 0, 1.0)
        veg_density_map[water_mask] = 0.0
        
        if ui_ready:
            im_bg_veg.set_data(veg_density_map)
            im_p1_veg.set_data(veg_density_map)
            im_p2_veg.set_data(veg_density_map)

        print(f"[TOPO] Fetching elevation...")
        lats = np.linspace(south, north, 8)
        lons = np.linspace(west, east, 8)
        grid_lats, grid_lons = np.meshgrid(lats, lons)
        lat_list = ",".join(map(str, grid_lats.flatten()))
        lon_list = ",".join(map(str, grid_lons.flatten()))
        
        topo_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_list}&longitude={lon_list}"
        
        try:
            res = requests.get(topo_url, timeout=15, verify=False)
            if res.status_code == 200:
                elev_data = res.json().get('elevation', [])
                if len(elev_data) == 64:
                    elevations = np.array(elev_data, dtype=np.float32).reshape(8, 8)
                    topo_img = Image.fromarray(elevations).resize((WIDTH, HEIGHT), Image.Resampling.BICUBIC)
                    elevation_map = np.array(topo_img)
                    
                    gy, gx = np.gradient(elevation_map)
                    slope_map = np.sqrt(gx**2 + gy**2)
                    slope_map = np.clip(slope_map / (slope_map.max() + 1e-6), 0, 1)
                    aspect_map = np.cos(np.arctan2(gy, gx))
                    
                    if ui_ready:
                        im_bg_topo.set_data(elevation_map)
                        im_bg_topo.set_clim(elevation_map.min(), elevation_map.max())
                        im_p1_topo.set_data(elevation_map)
                        im_p1_topo.set_clim(elevation_map.min(), elevation_map.max())
                        im_p2_topo.set_data(elevation_map)
                        im_p2_topo.set_clim(elevation_map.min(), elevation_map.max())
                    
                    print(f"[OK] Topography loaded")
        except Exception as e:
            print(f"[WARN] Topo error: {e}")

        if state.auto_mode:
            weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&daily=precipitation_sum&past_days=31&forecast_days=1"
            try:
                res = requests.get(weather_url, timeout=10, verify=False)
                if res.status_code == 200:
                    daily_rain = res.json().get('daily', {}).get('precipitation_sum', [])
                    days_since = 31
                    for i, rain in enumerate(reversed(daily_rain)):
                        if rain and rain > 1.0:
                            days_since = i
                            break
                    state.days_since_rain = days_since
                    state.drought_factor = np.clip(days_since / 20.0, 0.1, 1.0)
                    drought_map = np.full((HEIGHT, WIDTH), state.drought_factor, dtype=np.float32)
                    print(f"[OK] Drought: {days_since} days no rain")
            except:
                pass

        print(f"[OK] Map sync complete")
        if ui_ready:
            fig.canvas.draw_idle()
        return True
        
    except Exception as e:
        print(f"Map fetch error: {e}")
        return False

class MetalFireNN(nn.Module):
    def __init__(self, input_size=20):
        super().__init__()
        self.layers = [
            nn.Linear(input_size, 256),
            nn.Linear(256, 512),
            nn.Linear(512, 512),
            nn.Linear(512, 256),
            nn.Linear(256, 128),
            nn.Linear(128, 1)
        ]
    
    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = mx.maximum(x, 0)
        x = self.layers[-1](x)
        x = mx.sigmoid(x)
        return x

class FirePredictionModel:
    def __init__(self, model_path, stats_path):
        self.model_path = model_path
        self.stats_path = stats_path
        self.total_samples = 0
        self.model_available = False
        
        print(f"[MODEL] Loading {MODEL_INFO['name']} ({MODEL_INFO['parameters']:,} params)...")
        self.model = MetalFireNN()
        
        try:
            self.model.load_weights(model_path)
            self.model_available = True
            print(f"[OK] {MODEL_INFO['name']} loaded successfully!")
            print(f"[INFO] Architecture: {MODEL_INFO['layers']}")
            
            if stats_path and os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    stats = json.load(f)
                    self.total_samples = stats.get('total_samples', 0)
                    print(f"[OK] Training samples: {self.total_samples:,}")
        except Exception as e:
            print(f"\n{'='*70}")
            print(f"FATAL ERROR: Failed to load model!")
            print(f"{'='*70}")
            print(f"Error: {e}")
            print(f"Model path: {model_path}")
            print(f"{'='*70}\n")
            sys.exit(1)
    
    def predict_risk_map(self, params, elevation_map, slope_map, aspect_map,
                        drought_map, veg_density_map, confidence_threshold=0.7):
        
        wind_sin, wind_cos = encode_wind_angle(params['wind_angle'])
        tree_data = tree_types[params['tree_type']]
        season_data = seasons[params['season']]
        solar_rad = calculate_solar_radiation(params['hour'], params['season'])
        
        if params['temperature'] < 10.0:
            return np.zeros((HEIGHT, WIDTH)), np.zeros((HEIGHT, WIDTH))
        
        effective_temp = params['temperature'] + season_data['temp_mod'] * 10 + solar_rad * 5
        effective_humidity = np.clip(params['humidity'] + season_data['humidity_mod'], 0, 1)
        
        n_points = HEIGHT * WIDTH
        X = np.zeros((n_points, 20), dtype=np.float32)
        
        local_temps = effective_temp + aspect_map.flatten() * 3
        
        X[:, 0] = params['wind_strength']
        X[:, 1] = wind_sin
        X[:, 2] = wind_cos
        X[:, 3] = effective_humidity
        X[:, 4] = local_temps
        X[:, 5] = drought_map.flatten()
        X[:, 6] = (1.0 - tree_data['moisture'])
        X[:, 7] = tree_data['moisture']
        X[:, 8] = veg_density_map.flatten()
        X[:, 9] = elevation_map.flatten() / 1000.0
        X[:, 10] = slope_map.flatten()
        X[:, 11] = aspect_map.flatten()
        X[:, 12] = solar_rad
        X[:, 13] = params['hour'] / 24.0
        X[:, 14] = season_data['risk_mod']
        X[:, 15] = tree_data['density']
        X[:, 16] = params['temperature'] / 50.0
        X[:, 17] = state.days_since_rain / 31.0
        X[:, 18] = state.drought_factor
        X[:, 19] = veg_density_map.flatten() * tree_data['density']
        
        X_mx = mx.array(X)
        predictions_mx = self.model(X_mx)
        predictions = np.array(predictions_mx).flatten()
        
        risk_map = predictions.reshape(HEIGHT, WIDTH)
        
        min_risk = np.min(risk_map)
        max_risk = np.max(risk_map)
        if max_risk > min_risk:
            risk_map = (risk_map - min_risk) / (max_risk - min_risk)
        else:
            risk_map = np.zeros_like(risk_map)
        
        risk_map = np.clip(risk_map, 0, 1)
        thresholded_map = np.where(risk_map >= confidence_threshold, risk_map, 0)
        
        return risk_map, thresholded_map

prediction_model = FirePredictionModel(MODEL_PATH, STATS_PATH)

elevation_map, slope_map, aspect_map = generate_terrain()
drought_map = generate_drought_map()
veg_density_map = generate_vegetation_density()

wind_angle = 90.0
wind_strength = 0.6
temperature = 25.0
humidity = 0.25
hour = 12.0
confidence_threshold = 0.75
tree_list = list(tree_types.keys())
season_list = list(seasons.keys())

bg_modes = ['Vegetation', 'Topography', 'Satellite']

fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor('#0a0a0a')

try:
    fig.canvas.manager.full_screen_toggle()
except:
    pass

map_width = 0.28
map_height = 0.22
right_column_x = 0.68

ax_main = fig.add_axes([0.05, 0.35, 0.58, 0.58])
ax_main.set_axis_off()
ax_main.set_facecolor('#000000')

im_bg_veg = ax_main.imshow(veg_density_map, cmap='Greens', aspect='equal', alpha=0.6, interpolation='bilinear', zorder=1, visible=False)
im_bg_topo = ax_main.imshow(elevation_map, cmap='terrain', aspect='equal', alpha=0.6, interpolation='bilinear', zorder=1, visible=False)
satellite_img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
im_bg_satellite = ax_main.imshow(satellite_img, aspect='equal', alpha=0.8, interpolation='bilinear', zorder=1, visible=True)

risk_map = np.zeros((HEIGHT, WIDTH))
risk_cmap = mcolors.LinearSegmentedColormap.from_list(
    'risk', ['#00000000', '#003300', '#00ff00', '#ffff00', '#ff9900', '#ff0000', '#8b0000']
)
im_main = ax_main.imshow(risk_map, cmap=risk_cmap, aspect='equal', interpolation='bilinear', zorder=2)

propagation_cmap = mcolors.LinearSegmentedColormap.from_list(
    'propagation',
    [
        (0.0, '#00000000'),
        (0.3, '#FFD70088'),
        (0.5, '#FFA500AA'),
        (0.7, '#FF6347CC'),
        (1.0, '#DC143CFF'),
    ],
    N=256
)
propagation_map = np.zeros((HEIGHT, WIDTH))
im_propagation = ax_main.imshow(propagation_map, cmap=propagation_cmap, aspect='equal', 
                                interpolation='bilinear', zorder=5, alpha=0.0, visible=False,
                                vmin=0, vmax=1)

title_text = "AURA | AI-based Unified Risk Assessment"
ax_main.set_title(title_text, color='#00ff00', fontsize=14, weight='bold', pad=10)

ax_p1 = fig.add_axes([right_column_x, 0.68, map_width, map_height])
ax_p1.set_axis_off()
im_p1_topo = ax_p1.imshow(elevation_map, cmap='terrain', aspect='equal', visible=True)
im_p1_veg = ax_p1.imshow(veg_density_map, cmap='Greens', aspect='equal', visible=False)
satellite_p1 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
im_p1_sat = ax_p1.imshow(satellite_p1, aspect='equal', visible=False)
ax_p1.set_title("TOPOGRAPHY", color='#ffffff', fontsize=12, weight='bold')

ax_p2 = fig.add_axes([right_column_x, 0.38, map_width, map_height])
ax_p2.set_axis_off()
im_p2_veg = ax_p2.imshow(veg_density_map, cmap='Greens', aspect='equal', visible=True)
im_p2_topo = ax_p2.imshow(elevation_map, cmap='terrain', aspect='equal', visible=False)
satellite_p2 = np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
im_p2_sat = ax_p2.imshow(satellite_p2, aspect='equal', visible=False)
ax_p2.set_title("VEGETATION", color='#ffffff', fontsize=12, weight='bold')

ax_wind = fig.add_axes([0.01, 0.90, 0.08, 0.08])
ax_wind.set_axis_off()
ax_wind.set_aspect('equal')
wind_circle = Circle((0.5, 0.5), 0.4, edgecolor='#00ffff', facecolor='none', linewidth=2.5, transform=ax_wind.transAxes)
ax_wind.add_patch(wind_circle)

wind_arrow_length = 0.3
wind_rad = np.radians(wind_angle)
wind_arrow = ax_wind.arrow(0.5, 0.5,
                           wind_arrow_length * np.cos(wind_rad),
                           wind_arrow_length * np.sin(wind_rad),
                           color='#00ffff', width=0.03, head_width=0.08, head_length=0.1, 
                           transform=ax_wind.transAxes, zorder=11)

wind_text = ax_wind.text(1.1, 0.5, f"WIND {wind_angle:.0f}°", color='#00ffff', fontsize=10, 
                        ha='left', va='center', weight='bold', transform=ax_wind.transAxes)

ax_main.annotate('N', xy=(0.03, 0.95), xycoords='axes fraction', xytext=(0.03, 0.88),
                 arrowprops=dict(facecolor='#ffffff', width=2, headwidth=8, headlength=10),
                 color='#ffffff', weight='bold', ha='center', fontsize=12)

risk_stats_text = ax_main.text(0.5, -0.05, "Risk Analysis | Initializing...",
    color='#ffffff', fontsize=10, ha='center', va='top', weight='bold',
    transform=ax_main.transAxes,
    bbox=dict(facecolor='#000000', alpha=0.85, edgecolor='#ff0000', linewidth=2, pad=8))

hover_text = ax_main.text(0.5, -0.13, "", color='#ffffff', fontsize=10, ha='center', va='top', 
                         transform=ax_main.transAxes, weight='bold',
                         bbox=dict(facecolor='#000000', alpha=0.9, edgecolor='#00ff00', pad=5))

def on_mouse_move(event):
    if event.inaxes == ax_main and not state.textbox_active:
        ix = int(event.xdata) if event.xdata else -1
        iy = int(event.ydata) if event.ydata else -1
        if 0 <= ix < WIDTH and 0 <= iy < HEIGHT:
            risk_val = risk_map[iy, ix]
            patch_id = fire_patches[iy, ix]
            
            if risk_val > 0.4 and patch_id > 0:
                cluster_mask = (fire_patches == patch_id)
                cluster_area_pixels = np.sum(cluster_mask)
                area_m2 = calculate_real_area(cluster_area_pixels, state.current_lat, state.view_height)
                area_str = format_area(area_m2)
                fd = fractal_dimension(cluster_mask)
                complexity = (fd - 1.0)
                controllability = max(0, (1.0 - complexity) * (1.0 - risk_val)) * 100
                
                hover_text.set_text(
                    f"RISK: {risk_val:.2%} | FRACTAL: {fd:.3f} | "
                    f"AREA: {area_str} | CONTROL: {controllability:.1f}%"
                )
                hover_text.set_visible(True)
            elif risk_val > 0.4:
                hover_text.set_text(f"RISK: {risk_val:.2%}")
                hover_text.set_visible(True)
            else:
                hover_text.set_visible(False)
        else:
            hover_text.set_visible(False)
    else:
        hover_text.set_visible(False)
    fig.canvas.draw_idle()

def on_key_press(event):
    if event.key == ' ' and not state.textbox_active:
        show_prediction(None)
    elif event.key == 'escape':
        state.textbox_active = False

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
fig.canvas.mpl_connect('key_press_event', on_key_press)

def neon_button(pos, label, callback, color='#5f9ea0'):
    ax = plt.axes(pos, facecolor='#000000')
    btn = Button(ax, label, color='#0a0a0a', hovercolor=color)
    btn.label.set_color('#ffffff')
    btn.label.set_fontweight('bold')
    btn.label.set_fontsize(9)
    btn.ax.patch.set_edgecolor(color)
    btn.ax.patch.set_linewidth(2.5)
    
    def onclick(e):
        btn.ax.patch.set_facecolor(color)
        btn.ax.patch.set_alpha(0.5)
        fig.canvas.draw_idle()
        plt.pause(0.08)
        btn.ax.patch.set_facecolor('#0a0a0a')
        btn.ax.patch.set_alpha(1.0)
        fig.canvas.draw_idle()
        callback(e)
    
    btn.on_clicked(onclick)
    return btn

def update_wind_arrow():
    global wind_arrow
    wind_arrow.remove()
    wind_rad = np.radians(wind_angle)
    wind_arrow = ax_wind.arrow(0.5, 0.5,
                               wind_arrow_length * np.cos(wind_rad),
                               wind_arrow_length * np.sin(wind_rad),
                               color='#00ffff', width=0.03, head_width=0.08, head_length=0.1, 
                               transform=ax_wind.transAxes, zorder=11)
    wind_text.set_text(f"WIND {wind_angle:.0f}°")
    fig.canvas.draw_idle()

def apply_weather_data(weather):
    if not state.auto_mode:
        return
    
    global temperature, humidity, wind_strength, wind_angle, hour
    
    temperature = weather['temperature']
    humidity = weather['humidity']
    wind_strength = weather['wind_speed'] / 50.0
    wind_angle = weather['wind_direction']
    
    current_hour = fetch_current_time(state.current_lat, state.current_lon)
    hour = current_hour
    
    sl_temp.set_val(temperature)
    sl_hum.set_val(humidity * 100)
    sl_wind.set_val(wind_strength * 50)
    sl_angle.set_val(wind_angle)
    sl_hour.set_val(hour)
    
    update_wind_arrow()
    update_ui_info()
    
    print(f"[OK] Weather applied")

def show_prediction(e):
    global risk_map, fire_patches, propagation_map
    
    ax_main.set_title("PREDICTING...", color='#ffaa00', fontsize=16, weight='bold', pad=10)
    fig.canvas.draw_idle()
    
    if temperature < 10.0:
        risk_map.fill(0)
        im_main.set_data(risk_map)
        propagation_map.fill(0)
        im_propagation.set_data(propagation_map)
        for coll in ax_main.collections:
            coll.remove()
        
        title_with_model = (
            f"{MODEL_INFO['name']} | {tree_list[state.current_tree_idx]} | "
            f"{season_list[state.current_season_idx]} | {int(hour):02d}:00 | "
            f"{temperature:.0f}°C | Risk: 0.0%"
        )
        ax_main.set_title(title_with_model, color='#00ff00', fontsize=13, weight='bold', pad=10)
        fig.canvas.draw_idle()
        return
    
    params = {
        'wind_strength': wind_strength,
        'wind_angle': wind_angle,
        'humidity': humidity,
        'temperature': temperature,
        'hour': hour,
        'season': season_list[state.current_season_idx],
        'tree_type': tree_list[state.current_tree_idx]
    }
    
    risk_map_full, risk_map_thresh = prediction_model.predict_risk_map(
        params, elevation_map, slope_map, aspect_map,
        drought_map, veg_density_map, confidence_threshold
    )
    
    if water_mask is not None:
        risk_map_full[water_mask] = 0
        risk_map_thresh[water_mask] = 0
    
    if risk_map_thresh is not None:
        risk_map_thresh = median_filter(risk_map_thresh, size=3)
        risk_map_full = median_filter(risk_map_full, size=3)
        fire_patches, _ = label(risk_map_thresh > 0)
    else:
        fire_patches = np.zeros((HEIGHT, WIDTH))
    
    if risk_map_full is not None:
        noise = (np.random.rand(HEIGHT, WIDTH) * 0.1) - 0.05
        risk_map = np.clip(risk_map_full + noise, 0, 1)
        
        im_main.set_data(risk_map)
        
        tree_data = tree_types[tree_list[state.current_tree_idx]]
        propagation_map = calculate_propagation_potential(
            risk_map, wind_angle, wind_strength, slope_map, aspect_map,
            veg_density_map, tree_data['moisture']
        )
        im_propagation.set_data(propagation_map)
        
        if state.propagation_enabled:
            im_propagation.set_visible(True)
            im_propagation.set_alpha(0.7)
        else:
            im_propagation.set_visible(False)
        
        for coll in ax_main.collections:
            coll.remove()
        
        if np.max(risk_map_thresh) > 0:
            high_risk_levels = [0.7, 0.75, 0.8, 0.85]
            ax_main.contourf(risk_map_thresh, levels=high_risk_levels,
                           colors=['#ff9900', '#ff6600', '#ff3300'],
                           alpha=0.6, zorder=8)
            
            critical_risk = np.where(risk_map_thresh >= 0.85, risk_map_thresh, 0)
            if np.max(critical_risk) > 0:
                ax_main.contour(critical_risk, levels=[0.85],
                              colors=['#ff0000'], linewidths=3,
                              alpha=0.9, zorder=9)
                ax_main.contourf(critical_risk, levels=[0.85, 1.0],
                               colors=['#ff0000'], alpha=0.4, zorder=7)
        
        total_pixels = HEIGHT * WIDTH
        high_risk_pixels = np.sum(risk_map_thresh > 0.7)
        critical_pixels = np.sum(risk_map_thresh >= 0.85)
        high_risk_area = (high_risk_pixels / total_pixels) * 100
        critical_area = (critical_pixels / total_pixels) * 100
        
        avg_risk_high = np.mean(risk_map_thresh[risk_map_thresh > 0]) if high_risk_pixels > 0 else 0
        max_risk = np.max(risk_map_thresh)
        
        if state.propagation_enabled:
            prop_pixels = np.sum(propagation_map > 0.3)
            prop_area = (prop_pixels / total_pixels) * 100
            risk_stats_text.set_text(
                f"RISK | Critical: {critical_area:.1f}% | High: {high_risk_area:.1f}% | "
                f"Propagation: {prop_area:.1f}% | {state.current_date.strftime('%d/%m/%Y')}"
            )
        else:
            risk_stats_text.set_text(
                f"RISK | Critical: {critical_area:.1f}% | High: {high_risk_area:.1f}% | "
                f"Avg: {avg_risk_high:.1%} | Max: {max_risk:.1%} | {state.current_date.strftime('%d/%m/%Y')}"
            )
        
        title_with_model = (
            f"{MODEL_INFO['name']} | {tree_list[state.current_tree_idx]} | "
            f"{season_list[state.current_season_idx]} | {int(hour):02d}:00 | "
            f"{temperature:.0f}°C | Risk: {high_risk_area:.1f}%"
        )
        ax_main.set_title(title_with_model, color='#00ff00', fontsize=13, weight='bold', pad=10)
        
        fig.canvas.draw_idle()

def change_tree(e):
    state.current_tree_idx = (state.current_tree_idx + 1) % len(tree_list)
    btn_tree.label.set_text(f'TREE\n{tree_list[state.current_tree_idx]}')
    update_ui_info()
    show_prediction(None)

def change_season(e):
    state.current_season_idx = (state.current_season_idx + 1) % len(season_list)
    btn_season.label.set_text(f'SEASON\n{season_list[state.current_season_idx]}')
    update_ui_info()
    show_prediction(None)

def change_background(e):
    state.current_bg_mode = (state.current_bg_mode + 1) % len(bg_modes)
    
    for im in [im_bg_veg, im_bg_topo, im_bg_satellite, 
               im_p1_topo, im_p1_veg, im_p1_sat,
               im_p2_topo, im_p2_veg, im_p2_sat]:
        im.set_visible(False)
    
    if state.current_bg_mode == 0:
        im_bg_veg.set_visible(True)
        im_p1_topo.set_visible(True)
        im_p2_sat.set_visible(True)
    elif state.current_bg_mode == 1:
        im_bg_topo.set_visible(True)
        im_p1_veg.set_visible(True)
        im_p2_sat.set_visible(True)
    elif state.current_bg_mode == 2:
        im_bg_satellite.set_visible(True)
        im_p1_topo.set_visible(True)
        im_p2_veg.set_visible(True)
    
    btn_bg.label.set_text(f'BG\n{bg_modes[state.current_bg_mode]}')
    fig.canvas.draw_idle()

def exit_app(e):
    plt.close('all')
    os._exit(0)

def toggle_propagation(e):
    state.propagation_enabled = not state.propagation_enabled
    
    if state.propagation_enabled:
        btn_propagation.label.set_text('PROP\nON')
        btn_propagation.ax.patch.set_edgecolor('#ff6600')
        im_propagation.set_visible(True)
        im_propagation.set_alpha(0.7)
        print("[PROPAGATION] ON - Showing spread potential")
    else:
        btn_propagation.label.set_text('PROP\nOFF')
        btn_propagation.ax.patch.set_edgecolor('#666666')
        im_propagation.set_visible(False)
        print("[PROPAGATION] OFF")
    
    fig.canvas.draw_idle()

def update_ui_info():
    mode_str = "AUTO" if state.auto_mode else "MANUAL"
    model_type = MODEL_INFO['name']
    info_text.set_text(
        f"Model: {model_type} ({MODEL_INFO['parameters']:,} params) | {mode_str} | "
        f"{temperature:.0f}°C | {humidity*100:.0f}% | {int(hour):02d}:00 | "
        f"{season_list[state.current_season_idx]} | {tree_list[state.current_tree_idx]} | "
        f"{state.current_date.strftime('%d/%m/%Y')}"
    )
    fig.canvas.draw_idle()

btn_y = 0.22
btn_h = 0.05
btn_w = 0.08

btn_predict = neon_button([0.10, btn_y, btn_w, btn_h], 'PREDICT', show_prediction, '#00ffff')
btn_tree = neon_button([0.19, btn_y, btn_w, btn_h], f'TREE\n{tree_list[state.current_tree_idx]}', change_tree, '#228B22')
btn_season = neon_button([0.28, btn_y, btn_w, btn_h], f'SEASON\n{season_list[state.current_season_idx]}', change_season, '#FFD700')
btn_bg = neon_button([0.37, btn_y, btn_w, btn_h], f'BG\n{bg_modes[state.current_bg_mode]}', change_background, '#9370DB')
btn_propagation = neon_button([0.46, btn_y, btn_w, btn_h], 'PROP\nOFF', toggle_propagation, '#666666')

def create_hybrid_control(ax_pos, tb_pos, label, val_min, val_max, val_init, color, callback):
    ax_s = plt.axes(ax_pos, facecolor='#1a1a1a')
    sl = Slider(ax_s, label, val_min, val_max, valinit=val_init, color=color, valfmt="%1.0f")
    sl.label.set_color('#ffffff')
    sl.label.set_fontsize(9)
    sl.valtext.set_visible(False)
    
    ax_t = plt.axes(tb_pos, facecolor='#000000')
    for spine in ax_t.spines.values():
        spine.set_visible(False)
    tb = TextBox(ax_t, "", initial=str(int(val_init)), color='#000000')
    tb.text_disp.set_color(color)
    tb.text_disp.set_fontsize(10)
    tb.text_disp.set_fontweight('bold')
    tb.cursor.set_color('#ffffff')
    ax_t.patch.set_facecolor('#000000')
    
    updating = [False]
    
    def on_sl_change(v):
        if updating[0]: return
        updating[0] = True
        tb.set_val(f"{int(v)}")
        callback(v)
        updating[0] = False
    
    def on_tb_submit(t):
        if updating[0]: return
        try:
            v = float(t)
            v = np.clip(v, val_min, val_max)
            updating[0] = True
            sl.set_val(v)
            callback(v)
            updating[0] = False
        except: 
            updating[0] = False
    
    def on_tb_click(event):
        if event.inaxes == ax_t:
            state.textbox_active = True
    
    def on_tb_release(event):
        state.textbox_active = False
    
    sl.on_changed(on_sl_change)
    tb.on_submit(on_tb_submit)
    fig.canvas.mpl_connect('button_press_event', on_tb_click)
    fig.canvas.mpl_connect('button_release_event', on_tb_release)
    
    return sl, tb

s_x1, s_x2 = 0.12, 0.44
s_y = 0.14
s_w, s_h = 0.13, 0.015
t_w, t_h = 0.04, 0.03
t_off = 0.14

def update_pred(v=None):
    show_prediction(None)

def cb_wind(v): 
    global wind_strength
    wind_strength = v/50.0
    update_wind_arrow()
    update_pred()

def cb_angle(v): 
    global wind_angle
    wind_angle = v
    update_wind_arrow()
    update_pred()

def cb_temp(v): 
    global temperature
    temperature = v
    update_ui_info()
    update_pred()

def cb_hum(v): 
    global humidity
    humidity = v/100.0
    update_ui_info()
    update_pred()

def cb_hour(v): 
    global hour
    hour = v
    update_ui_info()
    update_pred()

def cb_conf(v): 
    global confidence_threshold
    confidence_threshold = v/100.0
    update_pred()

sl_wind, tb_wind = create_hybrid_control([s_x1, s_y, s_w, s_h], [s_x1 + t_off, s_y-0.008, t_w, t_h], 'Wind: ', 0, 100, 30, '#00ffff', cb_wind)
sl_angle, tb_angle = create_hybrid_control([s_x1, s_y - 0.03, s_w, s_h], [s_x1 + t_off, s_y - 0.038, t_w, t_h], 'Angle: ', 0, 360, 90, '#00ffff', cb_angle)
sl_temp, tb_temp = create_hybrid_control([s_x1, s_y - 0.06, s_w, s_h], [s_x1 + t_off, s_y - 0.068, t_w, t_h], 'Temp: ', -10, 50, 25, '#ff6600', cb_temp)
sl_hum, tb_hum = create_hybrid_control([s_x2, s_y, s_w, s_h], [s_x2 + t_off, s_y-0.008, t_w, t_h], 'Hum: ', 0, 100, 25, '#3399ff', cb_hum)
sl_hour, tb_hour = create_hybrid_control([s_x2, s_y - 0.03, s_w, s_h], [s_x2 + t_off, s_y - 0.038, t_w, t_h], 'Hour: ', 0, 23, 12, '#ffaa00', cb_hour)
sl_conf, tb_conf = create_hybrid_control([s_x2, s_y - 0.06, s_w, s_h], [s_x2 + t_off, s_y - 0.068, t_w, t_h], 'Conf: ', 50, 99, 75, '#ff00ff', cb_conf)

date_y = 0.28
date_h = 0.03
date_w = 0.05
right_column_x = 0.68

ax_day = plt.axes([right_column_x + 0.02, date_y, date_w, date_h], facecolor='#000000')
for spine in ax_day.spines.values():
    spine.set_visible(False)
input_day = TextBox(ax_day, 'Day: ', initial=str(state.current_date.day), color='#000000')
input_day.label.set_color('#ffffff')
input_day.text_disp.set_color('#00ff00')
input_day.cursor.set_color('#ffffff')
ax_day.patch.set_facecolor('#000000')

ax_month = plt.axes([right_column_x + 0.10, date_y, date_w, date_h], facecolor='#000000')
for spine in ax_month.spines.values():
    spine.set_visible(False)
input_month = TextBox(ax_month, 'Mon: ', initial=str(state.current_date.month), color='#000000')
input_month.label.set_color('#ffffff')
input_month.text_disp.set_color('#00ff00')
input_month.cursor.set_color('#ffffff')
ax_month.patch.set_facecolor('#000000')

ax_year = plt.axes([right_column_x + 0.18, date_y, date_w + 0.01, date_h], facecolor='#000000')
for spine in ax_year.spines.values():
    spine.set_visible(False)
input_year = TextBox(ax_year, 'Year: ', initial=str(state.current_date.year), color='#000000')
input_year.label.set_color('#ffffff')
input_year.text_disp.set_color('#00ff00')
input_year.cursor.set_color('#ffffff')
ax_year.patch.set_facecolor('#000000')

def on_date_change(text):
    try:
        day = int(input_day.text)
        month = int(input_month.text)
        year = int(input_year.text)
        
        new_date = datetime(year, month, day)
        state.current_date = new_date
        
        if state.auto_mode:
            season = get_season_from_date(new_date)
            state.current_season_idx = season_list.index(season)
            btn_season.label.set_text(f'SEASON\n{season}')
            
            weather = fetch_weather_data(state.current_lat, state.current_lon, new_date)
            if weather:
                apply_weather_data(weather)
        
        update_ui_info()
        show_prediction(None)
    except:
        pass

input_day.on_submit(on_date_change)
input_month.on_submit(on_date_change)
input_year.on_submit(on_date_change)

coord_x = right_column_x
start_y = 0.20

ax_lat = plt.axes([coord_x + 0.07, start_y, 0.06, 0.03], facecolor='#000000')
for spine in ax_lat.spines.values():
    spine.set_visible(False)
input_pt_lat = TextBox(ax_lat, 'LAT: ', initial=str(state.current_lat), color='#000000')
input_pt_lat.label.set_color('#00ff00')
input_pt_lat.text_disp.set_color('#00ff00')
input_pt_lat.cursor.set_color('#ffffff')
ax_lat.patch.set_facecolor('#000000')

ax_lon = plt.axes([coord_x + 0.18, start_y, 0.06, 0.03], facecolor='#000000')
for spine in ax_lon.spines.values():
    spine.set_visible(False)
input_pt_lon = TextBox(ax_lon, 'LON: ', initial=str(state.current_lon), color='#000000')
input_pt_lon.label.set_color('#00ff00')
input_pt_lon.text_disp.set_color('#00ff00')
input_pt_lon.cursor.set_color('#ffffff')
ax_lon.patch.set_facecolor('#000000')

ax_height = plt.axes([coord_x + 0.07, start_y - 0.045, 0.12, 0.03], facecolor='#000000')
for spine in ax_height.spines.values():
    spine.set_visible(False)
input_pt_height_m = TextBox(ax_height, 'HEIGHT (m): ', initial=str(int(state.view_height * 1000)), color='#000000')
input_pt_height_m.label.set_color('#ffaa00')
input_pt_height_m.text_disp.set_color('#00ff00')
input_pt_height_m.cursor.set_color('#ffffff')
ax_height.patch.set_facecolor('#000000')

def on_bbox_sync(e):
    try:
        new_lat = float(input_pt_lat.text)
        new_lon = float(input_pt_lon.text)
        new_height_km = float(input_pt_height_m.text) / 1000.0
        
        state.current_lat = new_lat
        state.current_lon = new_lon
        state.view_height = new_height_km
        
        fetch_map_data(new_lat, new_lon)
        
        if state.auto_mode:
            weather = fetch_weather_data(new_lat, new_lon, state.current_date)
            if weather:
                apply_weather_data(weather)
    except:
        pass

def on_textbox_focus(ax_list):
    def on_press(event):
        for ax in ax_list:
            if event.inaxes == ax:
                state.textbox_active = True
                return
    
    def on_release(event):
        state.textbox_active = False
    
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('button_release_event', on_release)

on_textbox_focus([ax_lat, ax_lon, ax_height, ax_day, ax_month, ax_year])

def toggle_mode(e):
    state.auto_mode = not state.auto_mode
    
    if state.auto_mode:
        btn_mode.label.set_text('AUTO')
        btn_mode.ax.patch.set_edgecolor('#00ff00')
        
        weather = fetch_weather_data(state.current_lat, state.current_lon, state.current_date)
        if weather:
            apply_weather_data(weather)
        
        season = get_season_from_date(state.current_date)
        state.current_season_idx = season_list.index(season)
        btn_season.label.set_text(f'SEASON\n{season}')
    else:
        btn_mode.label.set_text('MANUAL')
        btn_mode.ax.patch.set_edgecolor('#ff9900')
    
    update_ui_info()
    fig.canvas.draw_idle()

btn_mode = neon_button([right_column_x + 0.01, 0.33, 0.08, 0.04], 'AUTO', toggle_mode, '#00ff00')
btn_fetch = neon_button([0.72, 0.02, 0.18, 0.045], 'SYNC', on_bbox_sync, '#FFD700')
btn_exit = neon_button([0.91, 0.02, 0.07, 0.045], 'EXIT', exit_app, '#ff0000')

info_text = fig.text(0.34, 0.01,
    f"{MODEL_INFO['name']} | {MODEL_INFO['architecture']} | {MODEL_INFO['parameters']:,} params",
    color='#ffffff', fontsize=8, ha='center', va='center',
    bbox=dict(facecolor='#000000', alpha=0.85, edgecolor='#00ffff', linewidth=2, pad=8))

print("\n" + "="*60)
print(f"AURA {MODEL_INFO['name']} - Fire Risk Prediction System")
print("="*60)
print(f"Model: {MODEL_INFO['architecture']}")
print(f"Parameters: {MODEL_INFO['parameters']:,}")
print(f"Architecture: {MODEL_INFO['layers']}")
print(f"Type: {MODEL_INFO['type']}")
print("="*60)

def init_data_fetch():
    if state.auto_mode:
        weather = fetch_weather_data(state.current_lat, state.current_lon, state.current_date)
        if weather:
            apply_weather_data(weather)
        
        season = get_season_from_date(state.current_date)
        state.current_season_idx = season_list.index(season)
        btn_season.label.set_text(f'SEASON\n{season}')

    fetch_map_data(state.current_lat, state.current_lon)
    show_prediction(None)

fig.canvas.draw()
plt.pause(0.1)
init_data_fetch()

while state.running:
    plt.pause(0.1)

plt.close(fig)
