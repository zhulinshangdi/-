import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. 页面全局配置 (Academic Style)
# ==========================================
st.set_page_config(
    page_title="人体热调节与失温虚拟仿真系统",
    page_icon="❄️",
    layout="wide"
)

# 注入 CSS：修复 SVG 渲染，定义学术字体
st.markdown("""
<style>
    /* 全局背景与字体 */
    .stApp {
        background-color: #F8F9FA;
        font-family: "Arial", "Microsoft YaHei", sans-serif;
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: #1E293B;
        font-family: "Times New Roman", serif;
        font-weight: 700;
    }
    
    /* SVG 容器：强制白色背景，居中，阴影 */
    .svg-wrapper {
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
    
    /* 数据卡片 */
    .metric-box {
        background: white;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 10px;
    }
    
    /* 警报状态栏 */
    .alert-container {
        padding: 15px;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin-top: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
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
        self.mass = mass         # kg
        self.area = area         # m2
        self.basal_met = basal_met # W
        self.vaso_factor = vaso_factor # 血管收缩敏感度
        self.temp_skin = 33.0    # 初始皮温
        self.history = [33.0]    # 温度记录

def run_simulation(env_temp, wind_speed, clo, met, is_wet, duration=120):
    """
    运行120分钟的热力学仿真
    """
    # 1. 初始化人体节段 (参数源自 Fiala et al. 1999)
    segments = {
        "Head":  BodySegment("Head", "头部", 4.5, 0.14, 12.0, 0.1),
        "Trunk": BodySegment("Trunk", "躯干", 30.0, 0.55, 45.0, 0.1), # 躯干稳态强
        "Arms":  BodySegment("Arms", "手臂", 4.0, 0.26, 3.0, 0.8),
        "Hands": BodySegment("Hands", "手部", 0.4, 0.08, 0.5, 3.0), # 末端敏感度高
        "Legs":  BodySegment("Legs", "腿部", 12.0, 0.60, 8.0, 0.8),
        "Feet":  BodySegment("Feet", "脚部", 1.0, 0.14, 0.5, 3.0)  # 末端敏感度高
    }
    
    central_blood_temp = 37.0 # 核心血温初始值
    time_points = np.arange(0, duration + 1)
    
    # 风寒修正 (Osczevski 模型)
    v_eff = wind_speed if wind_speed < 5 else wind_speed * 0.6
    
    # 仿真循环 (步长 1分钟)
    for t in range(duration):
        total_blood_return_heat = 0
        total_met_heat = 0
        
        for key, seg in segments.items():
            # A. 产热 (Metabolism)
            act_factor = met
            # 手脚产热能力随运动增加不明显
            if key in ["Hands", "Feet", "Head"]: 
                act_factor = 1.0 + (met-1)*0.1
            
            q_met = seg.basal_met * act_factor
            total_met_heat += q_met
            
            # B. 散热 (Heat Loss)
            # 潮湿惩罚：热阻衰减至 35%
            real_clo = clo * 0.35 if is_wet else clo
            # 暴露部位修正
            if key in ["Head", "Hands"]: 
                local_clo = real_clo * 0.3
            else: 
                local_clo = real_clo
            
            r_total = 0.155 * local_clo + 0.1 / (1 + 0.5*v_eff)
            q_loss = seg.area * (seg.temp_skin - env_temp) / r_total
            
            # C. 血液灌注与逆流热交换 (Counter-current Exchange)
            # 核心机制：当核心温度 < 36.8°C，血管收缩启动
            vaso_response = 1.0
            if central_blood_temp < 36.8:
                delta = 36.8 - central_blood_temp
                # 敏感度越高，血流关闭越狠 (Hands/Feet vaso_factor=3.0)
                vaso_response = 1.0 / (1.0 + seg.vaso_factor * delta * 8.0)
            
            # 血液带来的热量 (W)
            q_blood = 18.0 * seg.mass * vaso_response * (central_blood_temp - seg.temp_skin) / 60.0
            
            # 计算回心血流的冷却效应
            total_blood_return_heat -= q_blood
            
            # D. 节点温度更新 (热容法)
            # Energy Balance: Q_net = Q_met + Q_blood - Q_loss
            net_heat_joules = (q_met + q_blood - q_loss) * 60 # 60秒
            dt = net_heat_joules / (seg.mass * 3470) # 人体比热容 3470
            
            seg.temp_skin += dt
            # 物理限制
            if seg.temp_skin < env_temp: seg.temp_skin = env_temp
            
            seg.history.append(seg.temp_skin)
            
        # 更新核心血温 (简化核心模型)
        core_mass = 50.0 
        # 核心温度变化 = (代谢产热 + 回心血热交换) / 热容
        core_dt = (total_met_heat * 1.5 + total_blood_return_heat) * 60 / (core_mass * 3470)
        central_blood_temp += core_dt
        
        # 生理稳态微调 (模拟寒战)
        if central_blood_temp < 37.0: central_blood_temp += 0.002
            
    return segments, time_points, central_blood_temp

# ==========================================
# 3. 可视化引擎：SVG 绘图 (无注释纯净版)
# ==========================================
def render_clean_svg(segments):
    # 颜色映射函数
    def get_color(t):
        if t < 10: return "#09090b" # 黑 (冻伤)
        if t < 20: return "#172554" # 深蓝
        if t < 28: return "#2563eb" # 蓝 (失温)
        if t < 33: return "#f59e0b" # 橙 (冷)
        return "#dc2626" # 红 (暖)

    cols = {k: get_color(v.history[-1]) for k, v in segments.items()}
    temps = {k: v.history[-1] for k, v in segments.items()}

    # 构建 SVG 字符串 (注意：不要添加 HTML 注释)
    svg = f"""
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
    return svg

# ==========================================
# 4. 主程序界面逻辑
# ==========================================

# --- 侧边栏：控制台 ---
st.sidebar.title("🎮 实验控制台 (Control)")
st.sidebar.markdown("---")

st.sidebar.subheader("1. 环境参数")
env_temp = st.sidebar.slider("环境温度 / Temp (°C)", -40, 15, -10)
wind_speed = st.sidebar.slider("风速 / Wind (km/h)", 0, 100, 20)

st.sidebar.subheader("2. 参与者状态")
met_val = st.sidebar.selectbox("代谢率 / Activity", [1.0, 3.0, 6.0, 8.0], format_func=lambda x: f"{x} METs (运动强度)")
clo_val = st.sidebar.slider("服装热阻 / Clothing (Clo)", 0.5, 4.0, 1.5, help="1.5=冲锋衣, 3.0=羽绒服")
is_wet = st.sidebar.checkbox("衣物湿透 / Wetness", False, help="模拟汗湿或落水，极度危险")

st.sidebar.markdown("---")
st.sidebar.caption("Model based on Fiala et al. (1999) & Huizenga (2001)")

# --- 运行计算 ---
segments, time_x, core_temp = run_simulation(env_temp, wind_speed, clo_val, met_val, is_wet)

# --- 主界面 ---
st.title("🏔️ 户外运动失温伤害虚拟仿真系统")
st.markdown("""
<div style='font-size: 16px; color: #475569; margin-bottom: 20px;'>
    <strong>仿真原理：</strong> 本系统基于 <em>Fiala 多节点生物热模型</em>，动态模拟人体在极端环境下的热调节过程。
    重点展示<b>“逆流热交换 (Counter-current Exchange)”</b>机制：即人体为了保护核心脏器，会通过血管收缩牺牲手脚温度。
</div>
""", unsafe_allow_html=True)

col_visual, col_data = st.columns([1, 1.5])

# --- 左列：可视化 ---
with col_visual:
    st.subheader("人体热成像模拟 (Thermography)")
    
    # SVG 渲染容器
    st.markdown(f"""
    <div class="svg-wrapper">
        {render_clean_svg(segments)}
    </div>
    """, unsafe_allow_html=True)
    
    # 核心体温状态栏
    status_bg = "#10B981" # Green
    status_msg = "核心体温正常 (Normal)"
    
    if core_temp < 32:
        status_bg = "#B91C1C" # Red
        status_msg = "⚠️ 严重失温 (Severe Hypothermia)"
    elif core_temp < 35:
        status_bg = "#EA580C" # Orange
        status_msg = "🛑 轻度失温 (Mild Hypothermia)"
    elif core_temp < 36.5:
        status_bg = "#F59E0B" # Yellow
        status_msg = "🥶 冷应激 (Cold Stress)"
        
    st.markdown(f"""
    <div class="alert-container" style="background-color: {status_bg};">
        <h3 style="margin:0; color:white;">{status_msg}</h3>
        <p style="margin:5px 0 0 0; font-size:1.2rem;">Core Temp: <strong>{core_temp:.1f} °C</strong></p>
    </div>
    """, unsafe_allow_html=True)

# --- 右列：数据分析 ---
with col_data:
    st.subheader("生理参数动态监测 (Data Monitoring)")
    
    # 1. 交互式图表 (Plotly)
    fig = go.Figure()
    
    # 绘制核心(躯干) vs 末端(手/脚)
    fig.add_trace(go.Scatter(
        x=time_x, y=segments['Trunk'].history, 
        name="躯干 (Core)", 
        line=dict(color="#F97316", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=time_x, y=segments['Hands'].history, 
        name="手部 (Hand)", 
        line=dict(color="#3B82F6", width=3)
    ))
    fig.add_trace(go.Scatter(
        x=time_x, y=segments['Feet'].history, 
        name="脚部 (Foot)", 
        line=dict(color="#1E3A8A", width=3)
    ))
    
    fig.update_layout(
        title="核心与外周温度分离现象 (Core-Shell Separation)",
        xaxis_title="暴露时间 (Minutes)",
        yaxis_title="皮肤温度 (°C)",
        template="plotly_white",
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 2. 详细数据表格
    st.subheader("实时数据面板 (Real-time Statistics)")
    
    table_data = []
    for k, v in segments.items():
        start_t = v.history[0]
        curr_t = v.history[-1]
        loss = start_t - curr_t
        
        status = "✅ 正常"
        if curr_t < 15: status = "❄️ 冻伤风险"
        elif curr_t < 28: status = "🔵 失温"
        
        table_data.append({
            "部位": v.name_cn,
            "初始 (°C)": f"{start_t:.1f}",
            "当前 (°C)": f"{curr_t:.1f}",
            "温降 (°C)": f"{loss:.1f}",
            "状态评估": status
        })
        
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # 3. 教学分析框
    trunk_temp = segments['Trunk'].history[-1]
    hand_temp = segments['Hands'].history[-1]
    
    st.info(f"""
    **🧪 实验现象分析：**
    当前仿真结果显示，躯干温度为 **{trunk_temp:.1f}°C**，而手部温度降至 **{hand_temp:.1f}°C**。
    这种巨大的温差（{(trunk_temp-hand_temp):.1f}°C）证实了人体在低温下会优先牺牲末端供血，以维持心脏和大脑的温度。
    """)
