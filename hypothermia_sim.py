import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import math

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="珠峰攀登热力学仿真 (Based on MENEX_HA)",
    page_icon="🏔️",
    layout="wide"
)

st.markdown("""
<style>
    .stApp { background-color: #F1F5F9; font-family: "Arial", sans-serif; }
    h1, h2, h3 { color: #0F172A; font-family: "Times New Roman", serif; font-weight: 700; }
    
    .kpi-card {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .kpi-title { font-size: 12px; color: #64748B; text-transform: uppercase; }
    .kpi-value { font-size: 20px; font-weight: bold; color: #1E293B; }
    
    .scenario-box {
        background-color: #e0f2fe; border: 1px solid #7dd3fc; 
        padding: 10px; border-radius: 5px; margin-bottom: 10px; color: #0c4a6e;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心物理引擎：MENEX_HA 模型 (Błażejczyk et al. 2024)
# ==========================================

class PhysiologyEngine:
    def __init__(self):
        # 基础参数
        self.segments = {
            "Head":  {"mass": 4.5, "area": 0.14, "vaso": 0.2, "solar_w": 0.3},
            "Trunk": {"mass": 30.0, "area": 0.55, "vaso": 0.1, "solar_w": 0.5},
            "Arms":  {"mass": 4.0, "area": 0.26, "vaso": 0.8, "solar_w": 0.2},
            "Hands": {"mass": 0.4, "area": 0.08, "vaso": 3.0, "solar_w": 0.0},
            "Legs":  {"mass": 12.0, "area": 0.60, "vaso": 0.8, "solar_w": 0.4},
            "Feet":  {"mass": 1.0, "area": 0.14, "vaso": 3.0, "solar_w": 0.0}
        }
        # 状态初始化
        self.state = {k: {"temp": 33.0, "hist": [33.0]} for k in self.segments.keys()}
        self.core_temp = 37.0
        self.history_core = [37.0]

    def calc_altitude_pressure(self, altitude_m):
        # 气压随海拔衰减公式 (hPa)
        return 1013.25 * (1 - 2.25577e-5 * altitude_m) ** 5.25588

    def calc_max_metabolism(self, altitude_m, has_o2_support):
        # 论文 Eq 7-10: 缺氧导致 VO2max 下降，从而限制最大产热
        # 海平面 VO2max 设为 57 ml/kg/min (训练有素的登山者)
        sea_level_vo2 = 57.0 
        
        # 氧气辅助修正 (Mask)
        effective_alt = altitude_m - 3000 if has_o2_support else altitude_m
        if effective_alt < 0: effective_alt = 0
        
        # 简化的海拔衰减系数 (approx data from paper)
        hypoxia_factor = 1.0
        if effective_alt > 1500:
            hypoxia_factor = 1.0 - (effective_alt - 1500) / 7500.0
        if hypoxia_factor < 0.2: hypoxia_factor = 0.2
        
        # 最大代谢率限制 (W)
        # 基准最大产热 ~ 1000W (高强度), 随缺氧下降
        max_met_w = 1000.0 * hypoxia_factor
        return max_met_w, hypoxia_factor

    def run_step(self, env, climber):
        # env: {temp, wind, altitude, solar_rad}
        # climber: {target_met, clo, is_wet, o2_support}
        
        ap = self.calc_altitude_pressure(env['altitude'])
        max_met, hypoxia = self.calc_max_metabolism(env['altitude'], climber['o2_support'])
        
        # 1. 实际代谢产热 (M) - 受生理极限限制
        # 用户设定的 METs * 基础代谢(约80W)
        target_w = climber['target_met'] * 80.0
        real_m = min(target_w, max_met) # 有心无力：想快走但缺氧走不动
        
        total_heat_loss = 0
        total_blood_cooling = 0
        
        # 2. 呼吸热损失 (Respiration Heat Loss) - Paper Eq 32
        # 高海拔过度通气 + 干燥空气 = 巨大热损
        # 简化估算: Q_res 正比于 M 和 (37 - T_air)
        # 高海拔系数：海拔越高，空气越干，呼吸量越大
        ventilation_factor = 1.0 + (env['altitude'] / 3000.0)
        q_res = 0.0015 * real_m * (37 - env['temp']) * ventilation_factor
        
        # 3. 计算各部位热平衡
        v_eff = env['wind'] * 0.6 if env['wind'] >= 5 else env['wind']
        
        for name, seg in self.segments.items():
            current_skin = self.state[name]['temp']
            
            # A. 太阳辐射收益 (Solar Gain) - Paper Eq 11
            # 只有部分面积受光，且受衣物遮挡
            # 简单模型：Radiation * Area * Absorptivity * (1/Clo)
            # 衣服越厚，辐射收益越难进入皮肤，但衣服表面会热(此处简化为直接收益)
            q_solar = env['solar_rad'] * seg['area'] * seg['solar_w'] * 0.4 
            
            # B. 对流与传导散热
            real_clo = climber['clo'] * 0.35 if climber['is_wet'] else climber['clo']
            # 头部手部修正
            if name in ["Head", "Hands"]: real_clo *= 0.3
            
            r_insulation = 0.155 * real_clo + 0.1 / (1 + 0.5 * v_eff)
            q_conv = seg['area'] * (current_skin - env['temp']) / r_insulation
            
            # C. 血液灌注 (逆流热交换)
            vaso = 1.0
            if self.core_temp < 36.8:
                delta = 36.8 - self.core_temp
                vaso = 1.0 / (1.0 + seg['vaso'] * delta * 10.0) # 敏感度极高
            
            q_blood = 18.0 * seg['mass'] * vaso * (self.core_temp - current_skin) / 60.0
            
            # 局部热平衡
            # 分配代谢热：躯干和腿分得多
            local_met_ratio = 0.1
            if name in ["Trunk", "Legs"]: local_met_ratio = 0.35
            q_local_met = real_m * local_met_ratio
            
            net_joules = (q_local_met + q_blood + q_solar - q_conv) * 60
            
            # 更新皮温
            dt = net_joules / (seg['mass'] * 3470)
            new_temp = current_skin + dt
            if new_temp < env['temp']: new_temp = env['temp']
            
            self.state[name]['temp'] = new_temp
            self.state[name]['hist'].append(new_temp)
            
            total_blood_cooling -= q_blood

        # 4. 更新核心温度
        # 核心 = 代谢产热 - 呼吸散热 - 血液冷却
        core_mass = 50.0
        # 太阳辐射对核心的直接影响较小，主要通过皮温传导
        core_net_joules = (real_m - q_res + total_blood_cooling) * 60
        core_dt = core_net_joules / (core_mass * 3470)
        
        self.core_temp += core_dt
        # 寒战补偿 (极弱，因为高海拔缺氧限制了寒战能力)
        if self.core_temp < 36.5: self.core_temp += 0.001 * hypoxia
            
        self.history_core.append(self.core_temp)
        
        return {
            "ap": ap,
            "real_m": real_m,
            "q_res": q_res,
            "hypoxia": hypoxia
        }

# ==========================================
# 3. 可视化组件 (SVG + Iframe)
# ==========================================
def render_avatar(state):
    def get_col(t):
        if t < 0: return "#000000"
        if t < 15: return "#1e1b4b"
        if t < 25: return "#1d4ed8"
        if t < 32: return "#60a5fa"
        if t < 35: return "#fbbf24"
        return "#ef4444"

    cols = {k: get_col(v['temp']) for k, v in state.items()}
    vals = {k: v['temp'] for k, v in state.items()}
    
    html = f"""
    <!DOCTYPE html>
    <body style="margin:0; background:#fff; display:flex; justify-content:center;">
    <svg width="280" height="520" viewBox="0 0 280 520">
        <defs>
            <linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#1e1b4b"/><stop offset="1" stop-color="#ef4444"/></linearGradient>
        </defs>
        
        <!-- Head -->
        <g><path d="M140,50 Q140,20 160,20 Q180,20 180,50 Q180,70 160,70 Q140,70 140,50 Z" fill="{cols['Head']}" stroke="#333" stroke-width="2"/>
        <text x="190" y="55" font-family="Arial" font-size="14" font-weight="bold">{vals['Head']:.1f}°</text></g>
        
        <!-- Trunk -->
        <g><path d="M130,70 L190,70 L200,200 L120,200 Z" fill="{cols['Trunk']}" stroke="#333" stroke-width="2"/>
        <text x="160" y="140" text-anchor="middle" fill="white" font-family="Arial" font-weight="bold">{vals['Trunk']:.1f}</text></g>
        
        <!-- Arms -->
        <path d="M130,70 L100,160 L120,170 L140,80 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
        <path d="M190,70 L220,160 L200,170 L180,80 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
        
        <!-- Hands -->
        <g><path d="M100,160 L90,190 L110,200 L120,170 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
        <path d="M220,160 L230,190 L210,200 L200,170 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
        <text x="10" y="190" font-family="Arial" font-size="14" font-weight="bold">{vals['Hands']:.1f}°</text>
        <line x1="90" y1="190" x2="50" y2="190" stroke="#666"/></g>
        
        <!-- Legs -->
        <path d="M120,200 L110,400 L150,400 L155,200 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
        <path d="M200,200 L210,400 L170,400 L165,200 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
        
        <!-- Feet -->
        <g><path d="M110,400 L100,430 L140,430 L150,400 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
        <path d="M210,400 L220,430 L180,430 L170,400 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
        <text x="230" y="435" font-family="Arial" font-size="14" font-weight="bold">{vals['Feet']:.1f}°</text></g>
        
        <rect x="40" y="480" width="200" height="10" fill="url(#g)" rx="5"/>
        <text x="40" y="505" font-size="10">Frozen</text><text x="240" y="505" font-size="10" text-anchor="end">Normal</text>
    </svg>
    </body>
    """
    return html

# ==========================================
# 4. 主程序逻辑
# ==========================================

# --- 侧边栏：场景与参数 ---
st.sidebar.title("🎮 仿真控制台")

# 场景预设 (基于论文 Case Studies)
preset = st.sidebar.selectbox("📚 典型场景预设", 
    ["自定义 (Custom)", "春季登顶 (Spring Summit)", "冬季登顶 (Winter Summit)", "紧急露宿 (Emergency Bivouac)"])

# 默认参数
def_alt, def_temp, def_wind, def_sol, def_met, def_clo, def_o2 = 5000, -10, 20, 800, 3.0, 1.5, False

if preset == "春季登顶 (Spring Summit)":
    # 论文数据: Ta -26C, Wind 16m/s, Solar High
    def_alt, def_temp, def_wind, def_sol, def_met, def_clo, def_o2 = 8848, -26, 16, 1000, 6.0, 3.5, True
    st.sidebar.info("📝 **场景描述：** 5月好天气窗口，高太阳辐射，使用氧气辅助。热平衡相对容易维持。")

elif preset == "冬季登顶 (Winter Summit)":
    # 论文数据: Ta -36C, Wind 36m/s (Winter average)
    def_alt, def_temp, def_wind, def_sol, def_met, def_clo, def_o2 = 8848, -36, 36, 600, 6.0, 4.0, True
    st.sidebar.warning("⚠️ **场景描述：** 12月严寒，极低气温+狂风。即使有氧气和厚衣服，失温风险也极高。")

elif preset == "紧急露宿 (Emergency Bivouac)":
    # 论文数据: No Tent, Night, Wind Chill
    def_alt, def_temp, def_wind, def_sol, def_met, def_clo, def_o2 = 8500, -30, 25, 0, 1.0, 3.5, False
    st.sidebar.error("☠️ **场景描述：** 8500m无帐篷过夜，无氧气，无太阳辐射，静止不动。死亡地带的生存挑战。")

st.sidebar.markdown("---")
st.sidebar.subheader("1. 环境因子 (Environment)")
alt = st.sidebar.slider("海拔高度 (m)", 0, 9000, def_alt, step=100, help="影响气压和含氧量")
temp = st.sidebar.slider("气温 (°C)", -50, 20, def_temp)
wind = st.sidebar.slider("风速 (km/h)", 0, 100, def_wind)
solar = st.sidebar.slider("太阳辐射 (W/m²)", 0, 1200, def_sol, help="夜间为0，晴朗雪地反射可达1000+")

st.sidebar.subheader("2. 攀登者状态 (Climber)")
met = st.sidebar.number_input("目标运动强度 (METs)", 0.8, 10.0, def_met)
clo = st.sidebar.slider("服装热阻 (Clo)", 0.5, 6.0, def_clo, help="连体羽绒服约 4-6 Clo")
o2_sup = st.sidebar.checkbox("使用氧气辅助 (O2 Support)", value=def_o2, help="缓解缺氧，提高产热能力")
is_wet = st.sidebar.checkbox("衣物受潮 (Wet)", False)

# --- 运行仿真 ---
engine = PhysiologyEngine()
duration = 120
env_params = {"temp": temp, "wind": wind, "altitude": alt, "solar_rad": solar}
climber_params = {"target_met": met, "clo": clo, "is_wet": is_wet, "o2_support": o2_sup}

metrics_log = []
for _ in range(duration):
    m = engine.run_step(env_params, climber_params)
    metrics_log.append(m)

# --- 主界面显示 ---
st.title("🏔️ 珠峰攀登体温调节仿真系统 (Ver 7.0)")
st.caption("Based on: Błażejczyk et al. (2024). Simulations of human heat balance during Mt. Everest summit attempts.")

# 1. 关键指标栏 (KPIs)
last_metric = metrics_log[-1]
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">环境气压 (Air Pressure)</div>
        <div class="kpi-value">{last_metric['ap']:.0f} hPa</div>
        <div style="font-size:12px; color:#64748B">海平面 ~1013 hPa</div>
    </div>""", unsafe_allow_html=True)
with col2:
    loss_ratio = (last_metric['q_res'] / last_metric['real_m']) * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">呼吸热流失 (Resp. Loss)</div>
        <div class="kpi-value" style="color:#DC2626">-{last_metric['q_res']:.1f} W</div>
        <div style="font-size:12px; color:#64748B">占总产热的 {loss_ratio:.1f}%</div>
    </div>""", unsafe_allow_html=True)
with col3:
    eff_percent = last_metric['hypoxia'] * 100
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-title">生理产热效能 (Hypoxia)</div>
        <div class="kpi-value">{eff_percent:.0f}%</div>
        <div style="font-size:12px; color:#64748B">受缺氧限制</div>
    </div>""", unsafe_allow_html=True)
with col4:
    core_t = engine.core_temp
    status = "✅ 正常" if core_t > 36.5 else ("⚠️ 失温" if core_t > 35 else "☠️ 极危")
    color = "#10B981" if core_t > 36.5 else "#EF4444"
    st.markdown(f"""
    <div class="kpi-card" style="border-left-color:{color}">
        <div class="kpi-title">核心体温 (Core Temp)</div>
        <div class="kpi-value" style="color:{color}">{core_t:.1f} °C</div>
        <div style="font-size:12px; color:#64748B">{status}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# 2. 核心交互区
c_vis, c_chart = st.columns([1, 2])

with c_vis:
    st.subheader("人体热成像 (Thermography)")
    components.html(render_avatar(engine.state), height=530)

with c_chart:
    st.subheader("多维生理数据监测")
    
    # 图表1: 核心与末端温度
    times = np.arange(duration)
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=times, y=engine.history_core, name="核心 (Core)", line=dict(color="#F97316", width=3)))
    fig1.add_trace(go.Scatter(x=times, y=engine.state['Hands']['hist'], name="手部 (Hand)", line=dict(color="#3B82F6", width=2)))
    fig1.add_trace(go.Scatter(x=times, y=engine.state['Feet']['hist'], name="脚部 (Foot)", line=dict(color="#1E3A8A", width=2)))
    fig1.update_layout(height=250, margin=dict(t=20, b=20, l=40, r=20), title="核心-外周温差监测", template="plotly_white")
    st.plotly_chart(fig1, use_container_width=True)
    
    # 图表2: 能量平衡分析 (堆叠面积图)
    # 展示产热 vs 呼吸流失
    prod_hist = [m['real_m'] for m in metrics_log]
    res_hist = [m['q_res'] for m in metrics_log]
    
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=times, y=prod_hist, name="实际产热 (M)", fill='tozeroy', line=dict(color="#10B981")))
    fig2.add_trace(go.Scatter(x=times, y=res_hist, name="呼吸热损 (Res)", fill='tozeroy', line=dict(color="#EF4444")))
    fig2.update_layout(height=250, margin=dict(t=20, b=20, l=40, r=20), title="能量代谢分析: 产热 vs 呼吸损耗", template="plotly_white")
    st.plotly_chart(fig2, use_container_width=True)

# 3. 教学分析
st.info("""
**💡 论文知识点解析 (Analysis):**
1. **海拔效应:** 试着拖动“海拔”滑块。你会发现气压下降导致 $VO_2max$ 降低，即便你将运动强度设为 8.0 METs，**“实际产热”** (绿色曲线) 也会被强制压低。这就是为什么在8000米以上很难靠运动产热来御寒。
2. **呼吸热损:** 观察红色区域。在极高海拔，空气稀薄且干燥，**呼吸热流失 (Respiration Loss)** 甚至可能占到总代谢热的 20%-30%。这是低海拔地区不具备的特征。
3. **紧急露宿:** 选择“紧急露宿”预设。在没有太阳辐射 ($R=0$) 且静止不动的情况下，体温会呈直线下滑，完美复现了论文中 *Emergency Night* 的致命风险。
""")
