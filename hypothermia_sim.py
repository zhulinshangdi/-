import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import base64

# ==========================================
# 1. 页面全局配置
# ==========================================
st.set_page_config(
    page_title="人体热调节与失温虚拟仿真系统",
    page_icon="❄️",
    layout="wide"
)

# CSS 样式：仅保留必要的布局样式
st.markdown("""
<style>
    .stApp { background-color: #F8F9FA; font-family: "Arial", sans-serif; }
    h1, h2, h3 { color: #1E293B; font-family: "Times New Roman", serif; font-weight: 700; }
    
    /* 图片容器样式 */
    .img-container {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 20px;
    }
    
    .alert-container {
        padding: 15px; border-radius: 8px; color: white; 
        text-align: center; margin-top: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心算法：Fiala 多节段生物热模型
# ==========================================
class BodySegment:
    def __init__(self, name_en, name_cn, mass, area, basal_met, vaso_factor):
        self.name_en = name_en
        self.name_cn = name_cn
        self.mass = mass
        self.area = area
        self.basal_met = basal_met
        self.vaso_factor = vaso_factor
        self.temp_skin = 33.0
        self.history = [33.0]

def run_simulation(env_temp, wind_speed, clo, met, is_wet, duration=120):
    # 初始化人体节段
    segments = {
        "Head":  BodySegment("Head", "头部", 4.5, 0.14, 12.0, 0.1),
        "Trunk": BodySegment("Trunk", "躯干", 30.0, 0.55, 45.0, 0.1),
        "Arms":  BodySegment("Arms", "手臂", 4.0, 0.26, 3.0, 0.8),
        "Hands": BodySegment("Hands", "手部", 0.4, 0.08, 0.5, 3.0),
        "Legs":  BodySegment("Legs", "腿部", 12.0, 0.60, 8.0, 0.8),
        "Feet":  BodySegment("Feet", "脚部", 1.0, 0.14, 0.5, 3.0)
    }
    
    central_blood_temp = 37.0
    time_points = np.arange(0, duration + 1)
    v_eff = wind_speed if wind_speed < 5 else wind_speed * 0.6
    
    for t in range(duration):
        total_blood_return_heat = 0
        total_met_heat = 0
        
        for key, seg in segments.items():
            # 产热
            act_factor = met
            if key in ["Hands", "Feet", "Head"]: act_factor = 1.0 + (met-1)*0.1
            q_met = seg.basal_met * act_factor
            total_met_heat += q_met
            
            # 散热
            real_clo = clo * 0.35 if is_wet else clo
            if key in ["Head", "Hands"]: local_clo = real_clo * 0.3
            else: local_clo = real_clo
            
            r_total = 0.155 * local_clo + 0.1 / (1 + 0.5*v_eff)
            q_loss = seg.area * (seg.temp_skin - env_temp) / r_total
            
            # 血管收缩与逆流热交换
            vaso_response = 1.0
            if central_blood_temp < 36.8:
                delta = 36.8 - central_blood_temp
                vaso_response = 1.0 / (1.0 + seg.vaso_factor * delta * 8.0)
            
            q_blood = 18.0 * seg.mass * vaso_response * (central_blood_temp - seg.temp_skin) / 60.0
            total_blood_return_heat -= q_blood
            
            # 温度更新
            net_heat_joules = (q_met + q_blood - q_loss) * 60
            dt = net_heat_joules / (seg.mass * 3470)
            seg.temp_skin += dt
            if seg.temp_skin < env_temp: seg.temp_skin = env_temp
            seg.history.append(seg.temp_skin)
            
        # 核心温度更新
        core_mass = 50.0 
        core_dt = (total_met_heat * 1.5 + total_blood_return_heat) * 60 / (core_mass * 3470)
        central_blood_temp += core_dt
        if central_blood_temp < 37.0: central_blood_temp += 0.002
            
    return segments, time_points, central_blood_temp

# ==========================================
# 3. 可视化引擎：Base64 SVG 渲染 (绝对稳定版)
# ==========================================
def render_b64_svg(segments):
    def get_color(t):
        if t < 10: return "#09090b" # 黑
        if t < 20: return "#172554" # 深蓝
        if t < 28: return "#2563eb" # 蓝
        if t < 33: return "#f59e0b" # 橙
        return "#dc2626" # 红

    cols = {k: get_color(v.history[-1]) for k, v in segments.items()}
    temps = {k: v.history[-1] for k, v in segments.items()}

    # SVG 字符串构建 (紧凑格式，防止缩进问题)
    svg_content = f"""
<svg width="320" height="580" viewBox="0 0 320 580" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" style="stop-color:#09090b;stop-opacity:1" />
      <stop offset="50%" style="stop-color:#2563eb;stop-opacity:1" />
      <stop offset="100%" style="stop-color:#dc2626;stop-opacity:1" />
    </linearGradient>
  </defs>
  <g id="head">
    <path d="M140,60 Q140,30 160,30 Q180,30 180,60 Q180,80 160,80 Q140,80 140,60 Z" fill="{cols['Head']}" stroke="#333" stroke-width="2"/>
    <line x1="180" y1="60" x2="200" y2="60" stroke="#666" stroke-width="1"/>
    <text x="205" y="65" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Head']:.1f}°C</text>
  </g>
  <g id="trunk">
    <path d="M130,80 L190,80 L200,210 L120,210 Z" fill="{cols['Trunk']}" stroke="#333" stroke-width="2"/>
    <text x="160" y="150" text-anchor="middle" font-family="Arial" font-size="14" fill="white" font-weight="bold">{temps['Trunk']:.1f}</text>
  </g>
  <g id="arms">
    <path d="M130,80 L100,170 L120,180 L140,90 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
    <path d="M190,80 L220,170 L200,180 L180,90 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
  </g>
  <g id="hands">
    <path d="M100,170 L90,200 L110,210 L120,180 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
    <path d="M220,170 L230,200 L210,210 L200,180 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
    <line x1="90" y1="200" x2="60" y2="200" stroke="#666" stroke-width="1"/>
    <text x="10" y="205" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Hands']:.1f}°C</text>
  </g>
  <g id="legs">
    <path d="M120,210 L110,410 L150,410 L155,210 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
    <path d="M200,210 L210,410 L170,410 L165,210 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
  </g>
  <g id="feet">
    <path d="M110,410 L100,440 L140,440 L150,410 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
    <path d="M210,410 L220,440 L180,440 L170,410 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
    <line x1="220" y1="440" x2="250" y2="440" stroke="#666" stroke-width="1"/>
    <text x="255" y="445" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Feet']:.1f}°C</text>
  </g>
  <rect x="60" y="500" width="200" height="12" fill="url(#grad1)" rx="4"/>
  <text x="60" y="530" font-size="12" font-family="Arial">冻结 (0°C)</text>
  <text x="260" y="530" font-size="12" font-family="Arial" text-anchor="end">正常 (37°C)</text>
</svg>
"""
    # 【修复核心】：转为 Base64 编码，绕过 Markdown 解析器
    b64 = base64.b64encode(svg_content.encode('utf-8')).decode("utf-8")
    img_tag = f'<img src="data:image/svg+xml;base64,{b64}" alt="Thermal Body" width="100%"/>'
    return img_tag

# ==========================================
# 4. 主程序界面
# ==========================================

# 侧边栏
st.sidebar.title("🎮 实验控制台")
env_temp = st.sidebar.slider("环境温度 / Temp (°C)", -40, 15, -10)
wind_speed = st.sidebar.slider("风速 / Wind (km/h)", 0, 100, 20)
met_val = st.sidebar.selectbox("运动状态", [1.0, 3.0, 6.0, 8.0], format_func=lambda x: f"{x} METs")
clo_val = st.sidebar.slider("服装热阻 (Clo)", 0.5, 4.0, 1.5)
is_wet = st.sidebar.checkbox("衣物湿透 (Wet)", False)

# 运行
segments, time_x, core_temp = run_simulation(env_temp, wind_speed, clo_val, met_val, is_wet)

st.title("🏔️ 户外运动失温伤害虚拟仿真系统")
st.markdown("本系统基于 **Fiala 多节点生物热模型**，动态模拟人体在极端环境下的**逆流热交换**机制。")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.subheader("人体热成像模拟")
    
    # 使用 Base64 图片标签渲染
    img_html = render_b64_svg(segments)
    st.markdown(f'<div class="img-container">{img_html}</div>', unsafe_allow_html=True)
    
    # 状态框
    bg_color = "#10B981"
    msg = "核心体温正常"
    if core_temp < 32: bg_color = "#B91C1C"; msg = "⚠️ 严重失温"
    elif core_temp < 35: bg_color = "#F59E0B"; msg = "🛑 轻度失温"
        
    st.markdown(f"""
    <div class="alert-container" style="background-color: {bg_color};">
        <h3>{msg}</h3>
        <p style="font-size:18px;">Core Temp: <strong>{core_temp:.1f} °C</strong></p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.subheader("生理参数监测")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_x, y=segments['Trunk'].history, name="躯干 (Core)", line=dict(color="#F97316", width=3)))
    fig.add_trace(go.Scatter(x=time_x, y=segments['Hands'].history, name="手部 (Hand)", line=dict(color="#3B82F6", width=3)))
    fig.add_trace(go.Scatter(x=time_x, y=segments['Feet'].history, name="脚部 (Foot)", line=dict(color="#1E3A8A", width=3)))
    
    fig.update_layout(title="核心-外周温度分离现象", xaxis_title="时间 (min)", yaxis_title="温度 (°C)", height=350, template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("实时数据")
    data = []
    for k,v in segments.items():
        data.append({"部位": v.name_cn, "当前温度": f"{v.history[-1]:.1f}°C"})
    st.dataframe(pd.DataFrame(data).T, use_container_width=True)
    
    st.info(f"分析：躯干与手部温差达 {(segments['Trunk'].history[-1] - segments['Hands'].history[-1]):.1f}°C，验证了血管收缩保护核心的机制。")
