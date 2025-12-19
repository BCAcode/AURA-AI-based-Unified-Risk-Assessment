import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.widgets import Button, Slider, TextBox
from matplotlib.patches import Circle, FancyArrow
import matplotlib as mpl

mpl.rcParams['toolbar'] = 'None'
plt.style.use('dark_background')

HEIGHT = 144
WIDTH = int(HEIGHT * 16 / 9)
p_base = 0.3

EMPTY = 0
TREE = 1
BURNING = 2

tree_types = {
    "Çam":      {"spread_factor": 0.7, "color": "#228B22"},
    "Meşe":     {"spread_factor": 0.5, "color": "#006400"},
    "Çınar":    {"spread_factor": 1.2, "color": "#00ff7f"},
    "Kayın":    {"spread_factor": 0.6, "color": "#2e8b57"},
    "Kavak":    {"spread_factor": 1.3, "color": "#7fff00"},
    "Ardıç":    {"spread_factor": 0.9, "color": "#3cb371"},
    "Zeytin":   {"spread_factor": 0.8, "color": "#556b2f"},
    "Akasya":   {"spread_factor": 1.0, "color": "#66cdaa"},
    "Maki":     {"spread_factor": 1.4, "color": "#32cd32"},
    "Çalı":     {"spread_factor": 1.6, "color": "#adff2f"},
    "Karaışık": {"spread_factor": 1.0, "color": "#9acd32"}
}
current_tree_type = "Çam"

def init_forest():
    f = np.ones((HEIGHT, WIDTH), dtype=int) * TREE
    fuel = np.ones((HEIGHT, WIDTH))
    humidity = np.random.uniform(10, 40, (HEIGHT, WIDTH)) / 100
    temperature = np.full((HEIGHT, WIDTH), 25.0)
    f[HEIGHT // 2, WIDTH // 2] = BURNING
    return f, fuel, humidity, temperature

forest, fuel, humidity, temperature = init_forest()

def make_cmap_for_tree(tree_name):
    return mcolors.ListedColormap(
        ['#111111', tree_types[tree_name]["color"], '#ff8c00', '#ff0000']
    )

cmap = make_cmap_for_tree(current_tree_type)
bounds = [0, 1, 2, 3, 4]
norm = mcolors.BoundaryNorm(bounds, cmap.N)

wind_angle = 90.0
wind_strength = 0.3
drought_level = 0.0
simulation_started = False
tracking = False
running = True

fig, ax = plt.subplots(figsize=(16, 9))
try:
    fig.canvas.manager.set_window_title("AURA | AI-based Unified Risk Assessment")
except Exception:
    pass

ax.set_position([0.25, 0.45, 0.5, 0.375])
im = ax.imshow(forest, cmap=cmap, norm=norm, aspect='equal')
ax.set_axis_off()

fd_text = ax.text(
    WIDTH-50, 10, "Fraktal Değer: 0.00", color='white',
    fontsize=10, ha='right', va='top',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
)
control_text = ax.text(
    WIDTH-50, 26, "Kontrol Edilebilirlik: 0.00", color='white',
    fontsize=10, ha='right', va='top',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
)
firecov_text = ax.text(
    WIDTH-50, 42, "Yangın Yayılımı: 0.00", color='white',
    fontsize=10, ha='right', va='top',
    bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
)

wheel_ax = plt.axes([0.01,0.05,0.12,0.3], facecolor='black')
wheel_ax.set_aspect('equal')
wheel_ax.axis('off')
wheel_circle = Circle((0.5,0.5),0.45, edgecolor='#5f9ea0',
                      linewidth=2, facecolor='#111111')
wheel_ax.add_patch(wheel_circle)
line, = wheel_ax.plot([0.5,0.5],[0.5,0.9], color='#5f9ea0', lw=3)
wheel_ax.text(0.5, -0.08, "Rüzgar", color='white', fontsize=10,
              ha='center', va='top', transform=wheel_ax.transAxes)

def update_line(xdata, ydata):
    global wind_angle
    dx = xdata - 0.5
    dy = ydata - 0.5
    visual_angle = (np.degrees(np.arctan2(dy, dx))) % 360
    wind_angle = visual_angle
    line.set_data([0.5, 0.5 + 0.45*np.cos(np.radians(visual_angle))],
                  [0.5, 0.5 + 0.45*np.sin(np.radians(visual_angle))])
    fig.canvas.draw_idle()

def on_wheel_click(event):
    global tracking
    if event.inaxes == wheel_ax:
        tracking = not tracking

def on_wheel_motion(event):
    if tracking and event.inaxes == wheel_ax and event.xdata is not None and event.ydata is not None:
        update_line(event.xdata, event.ydata)

wheel_ax.figure.canvas.mpl_connect('button_press_event', on_wheel_click)
wheel_ax.figure.canvas.mpl_connect('motion_notify_event', on_wheel_motion)

tree_ax = plt.axes([0.01,0.36,0.12,0.58], facecolor='black')
tree_ax.set_xlim(0, 1)
tree_ax.set_ylim(0, 1)
tree_ax.set_aspect("equal", "box")
tree_ax.axis('off')

n_types = len(tree_types)
y_top = 0.95
y_bottom = 0.05
step = (y_top - y_bottom) / (n_types - 1) if n_types > 1 else 0
tree_positions = []
for i, name in enumerate(tree_types.keys()):
    y = y_top - i*step
    tree_positions.append((name, 0.16, y))

tree_circle_patches = {}
circle_radius = 0.03

for name, cx, cy in tree_positions:
    circ = Circle((cx, cy), circle_radius,
                  edgecolor=tree_types[name]["color"],
                  facecolor='none', linewidth=1.6, zorder=5)
    tree_ax.add_patch(circ)
    tree_ax.text(cx+0.12, cy, name, color='white',
                 fontsize=9, va='center', ha='left')
    tree_circle_patches[name] = circ

def update_tree_selector_visuals(selected_name):
    for nm, circ in tree_circle_patches.items():
        if nm == selected_name:
            circ.set_facecolor(tree_types[nm]["color"])
            circ.set_edgecolor('#ffffff')
            circ.set_linewidth(2.2)
        else:
            circ.set_facecolor('none')
            circ.set_edgecolor(tree_types[nm]["color"])
            circ.set_linewidth(1.4)
    fig.canvas.draw_idle()

def set_tree_type(tree_name):
    global current_tree_type, cmap, norm
    current_tree_type = tree_name
    cmap = make_cmap_for_tree(current_tree_type)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    im.set_cmap(cmap)
    im.set_norm(norm)
    update_tree_selector_visuals(current_tree_type)
    fig.suptitle(
        f"Ağaç tipi: {current_tree_type} | spread={tree_types[current_tree_type]['spread_factor']:.2f}",
        color='white', fontsize=10
    )
    fig.canvas.draw_idle()

update_tree_selector_visuals(current_tree_type)

def on_tree_ax_click(event):
    if event.inaxes != tree_ax or event.xdata is None or event.ydata is None:
        return
    for name, cx, cy in tree_positions:
        dx = event.xdata - cx
        dy = event.ydata - cy
        if np.hypot(dx, dy) <= circle_radius:
            set_tree_type(name)
            break

fig.canvas.mpl_connect('button_press_event', on_tree_ax_click)

def neon_button(ax, label, callback):
    btn = Button(ax, label, color='black', hovercolor='#5f9ea0')
    btn.label.set_color('white')
    btn.label.set_fontweight('bold')
    btn.ax.patch.set_edgecolor('#5f9ea0')
    btn.ax.patch.set_facecolor('none')
    def onclick(event):
        btn.ax.patch.set_facecolor('white')
        fig.canvas.draw_idle()
        plt.pause(0.08)
        btn.ax.patch.set_facecolor('none')
        fig.canvas.draw_idle()
        callback(event)
    btn.on_clicked(onclick)
    return btn

start_ax = plt.axes([0.53,0.05,0.1,0.07], facecolor='black')
button_start = neon_button(start_ax,'BAŞLA', lambda e: globals().update({'simulation_started': True}))

stop_ax = plt.axes([0.65,0.05,0.1,0.07], facecolor='black')
button_stop = neon_button(stop_ax,'DUR', lambda e: globals().update({'simulation_started': False}))

reset_ax = plt.axes([0.77,0.05,0.1,0.07], facecolor='black')
def reset_sim(event):
    global forest, fuel, humidity, temperature, simulation_started
    forest, fuel, humidity, temperature = init_forest()
    simulation_started = False
    im.set_data(forest)
button_reset = neon_button(reset_ax,'SIFIRLA', reset_sim)

exit_ax = plt.axes([0.89,0.05,0.1,0.07], facecolor='black')
button_exit = neon_button(exit_ax,'ÇIKIŞ', lambda e: globals().update({'running': False}))

def add_slider_with_textbox(ax_slider, ax_text, label, valmin, valmax, valinit, callback_slider, callback_text):
    slider = Slider(ax_slider, label, valmin, valmax, valinit=valinit, color='#5f9ea0')
    
    textbox = TextBox(ax_text, '', initial=str(valinit), color='black', hovercolor='#111111')
    textbox.text_disp.set_color('white')
    textbox._editing = False

    def on_text_click(event):
        if event.inaxes == ax_text:
            textbox._editing = True
            ax_text.set_facecolor('#111111')
            fig.canvas.draw_idle()
    def on_text_release(event):
        if textbox._editing:
            textbox._editing = False
            ax_text.set_facecolor('#111111')
            fig.canvas.draw_idle()
    fig.canvas.mpl_connect('button_press_event', on_text_click)
    fig.canvas.mpl_connect('button_release_event', on_text_release)

    def slider_update(val):
        textbox.set_val(f"{val:.2f}")
        callback_slider(val)
    slider.on_changed(slider_update)

    def text_update(text):
        try:
            val = float(text)
            val = np.clip(val, valmin, valmax)
            slider.set_val(val)
            callback_text(val)
        except ValueError:
            pass
    textbox.on_submit(text_update)

    return slider, textbox

strength_ax = plt.axes([0.25,0.25,0.45,0.03], facecolor='#111111')
strength_text_ax = plt.axes([0.71,0.25,0.05,0.03], facecolor='#111111')
strength_slider, strength_text = add_slider_with_textbox(
    strength_ax, strength_text_ax, 'Rüzgar Hızı (km/h)', 0, 100, 30,
    lambda val: globals().update({'wind_strength': val / 50}),
    lambda val: globals().update({'wind_strength': val / 50})
)

humidity_ax = plt.axes([0.25,0.3,0.45,0.03], facecolor='#111111')
humidity_text_ax = plt.axes([0.71,0.3,0.05,0.03], facecolor='#111111')
humidity_slider, humidity_text = add_slider_with_textbox(
    humidity_ax, humidity_text_ax, 'Nem %', 0, 100, 25,
    lambda val: globals().update({'humidity': np.full((HEIGHT, WIDTH), val/100)}),
    lambda val: globals().update({'humidity': np.full((HEIGHT, WIDTH), val/100)})
)

temp_ax = plt.axes([0.25,0.35,0.45,0.03], facecolor='#111111')
temp_text_ax = plt.axes([0.71,0.35,0.05,0.03], facecolor='#111111')
temp_slider, temp_text = add_slider_with_textbox(
    temp_ax, temp_text_ax, 'Sıcaklık (°C)', -40, 60, 25,
    lambda val: globals().update({'temperature': np.full((HEIGHT, WIDTH), val)}),
    lambda val: globals().update({'temperature': np.full((HEIGHT, WIDTH), val)})
)

drought_ax = plt.axes([0.25,0.4,0.45,0.03], facecolor='#111111')
drought_text_ax = plt.axes([0.71,0.4,0.05,0.03], facecolor='#111111')
drought_slider, drought_text = add_slider_with_textbox(
    drought_ax, drought_text_ax, 'Kuraklık %', 0, 100, 0,
    lambda val: globals().update({'drought_level': val/100}),
    lambda val: globals().update({'drought_level': val/100})
)

def fractal_dimension(Z, threshold=0.5):
    """Box-counting ile fraktal boyut hesapla"""
    Z = (Z > threshold)
    sizes = 2**np.arange(1, int(np.log2(min(Z.shape))), 1)
    counts = []
    for size in sizes:
        S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
                                   np.arange(0, Z.shape[1], size), axis=1)
        counts.append(np.count_nonzero(S))
    if len(counts) < 2:
        return 0.0
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]

def compute_controllability(fd, fuel_map, humidity_map, temp_map, wind_strength, drought_level, spread_factor, fire_coverage):
    avg_fuel = np.clip(np.mean(fuel_map), 0.0, 1.0)
    avg_humidity = np.clip(np.mean(humidity_map), 0.0, 1.0)
    avg_temp = np.mean(temp_map)

    hum_score = avg_humidity
    fuel_score = 1.0 - avg_fuel
    temp_score = 1.0 - np.clip((avg_temp - 20.0) / 40.0, -0.5, 1.0)
    temp_score = np.clip(temp_score, 0.0, 1.0)
    wind_norm = np.clip(wind_strength / 1.5, 0.0, 1.0)
    wind_score = 1.0 - wind_norm
    drought_score = 1.0 - np.clip(drought_level, 0.0, 1.0)
    sf = spread_factor
    sf_norm = (sf - 0.4) / (1.6 - 0.4)
    sf_score = 1.0 - np.clip(sf_norm, 0.0, 1.0)

    fd_clamped = np.clip(fd, 0.5, 3.0)
    complexity = (fd_clamped - 1.0) / 2.0
    complexity = np.clip(complexity, 0.0, 1.0)
    complexity_score = 1.0 - complexity

    combined = (0.30 * hum_score
                + 0.20 * wind_score
                + 0.15 * fuel_score
                + 0.15 * temp_score
                + 0.10 * drought_score
                + 0.05 * sf_score
                + 0.05 * complexity_score)
    combined = np.clip(combined, 0.0, 1.0)

    base_controllability = combined * (1.0 - 0.6 * complexity)
    base_controllability = np.clip(base_controllability, 0.0, 1.0)

    k = 0.95  
    gamma = 0.5  
    coverage_penalty = np.clip(k * (fire_coverage ** gamma), 0.0, 0.99)

    controllability = base_controllability * (1.0 - coverage_penalty)
    controllability = np.clip(controllability, 0.0, 1.0)

    return controllability

def compute_burning(forest, fuel, humidity, temperature, wind_angle, wind_strength, fractal_factor, control_effectiveness=0.0):
    new_forest = forest.copy()
    wx_unit = np.cos(np.radians(wind_angle))
    wy_unit = np.sin(np.radians(wind_angle))
    spread_factor = tree_types[current_tree_type]["spread_factor"]
    
    for i in range(forest.shape[0]):
        for j in range(forest.shape[1]):
            if forest[i,j] != TREE or fuel[i,j] <= 0:
                continue
            burning_neighbors = 0.0
            for xi in [-1,0,1]:
                for yj in [-1,0,1]:
                    if xi == 0 and yj == 0:
                        continue
                    ni, nj = i+xi, j+yj
                    if 0 <= ni < forest.shape[0] and 0 <= nj < forest.shape[1] and forest[ni,nj] == BURNING:
                        neighbor_vec_cart = np.array([-yj, xi], dtype=float)
                        norm_vec = np.linalg.norm(neighbor_vec_cart)
                        if norm_vec > 0:
                            neighbor_unit = neighbor_vec_cart / norm_vec
                            dot = wx_unit * neighbor_unit[0] + wy_unit * neighbor_unit[1]
                            wind_alignment = max(0.0, dot)
                            opposite_alignment = max(0.0, -dot)
                            wind_effect = (1.0 + wind_strength*wind_alignment) * (1.0 - 0.98*opposite_alignment)
                            burning_neighbors += max(0.01, wind_effect)
            temp_factor = 1.0 + (temperature[i,j] - 20.0) / 50.0
            humidity_factor = 1.0 - 0.5 * humidity[i,j]
            prob = (1.0 - (1.0 - p_base) ** burning_neighbors) * fuel[i,j] * humidity_factor * temp_factor
            prob *= fractal_factor * spread_factor
            prob *= (1.0 + drought_level)
            if control_effectiveness > 0.0:
                suppression = np.clip(control_effectiveness * 0.7, 0.0, 0.95)
                prob *= (1.0 - suppression)
            prob = np.clip(prob, 0.0, 1.0)
            if np.random.rand() < prob:
                new_forest[i,j] = BURNING
    return new_forest

def draw_compass(ax):
    x = 15
    y = HEIGHT - 15
    ax.add_patch(FancyArrow(x, y, 0, -12, width=2,
                            head_width=6, head_length=6,
                            color='white', fill=False, zorder=9))
    ax.add_patch(FancyArrow(x, y, 0, -12, width=1.5,
                            head_width=5, head_length=5,
                            color='black', zorder=10))
    ax.text(x, y-18, 'K', color='white', fontsize=11,
            fontweight='bold', ha='center', va='bottom', zorder=10)

def draw_scale(ax, px_length=50, label="100 m"):
    y = HEIGHT - 10
    x_start = 10
    x_end = x_start + px_length
    ax.plot([x_start, x_end], [y, y], color='white', linewidth=3)
    ax.text((x_start+x_end)/2, y+5, label, color='white',
            ha='center', va='bottom', fontsize=5, fontweight='bold')

draw_compass(ax)
draw_scale(ax)
update_line(0.5, 0.9)

while running:
    plt.pause(0.05)
    if simulation_started:
        fd = fractal_dimension(forest != TREE)
        fractal_factor = np.clip(1.0 + (fd - 1.0) * 0.4, 0.5, 2.0)
        fd_text.set_text(f"FD: {fd:.2f}")

        fire_coverage = float(np.count_nonzero(forest == BURNING)) / (forest.size)
        firecov_text.set_text(f"Yangın Yayılımı: {fire_coverage:.3f}")

        controllability = compute_controllability(
            fd, fuel, humidity, temperature, wind_strength, drought_level,
            tree_types[current_tree_type]["spread_factor"], fire_coverage
        )
        control_text.set_text(f"Kontrol Edilebilirlik: {controllability:.2f}")

        control_effectiveness = controllability**1.5

        forest = compute_burning(forest, fuel, humidity, temperature, wind_angle, wind_strength, fractal_factor, control_effectiveness=control_effectiveness)

        burning_cells = (forest == BURNING)
        fuel[burning_cells] -= 0.1
        low_fuel_mask = burning_cells & (fuel <= 0)
        fuel[low_fuel_mask] = 0.0
        forest[low_fuel_mask] = EMPTY

        im.set_data(forest)
        fig.canvas.draw_idle()

        if not np.any(forest == BURNING):
            simulation_started = False

plt.close(fig)