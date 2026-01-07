import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ==========================================
# 1. 页面配置与学术风格定义 (Rigorous Style)
# ==========================================
st.set_page_config(
    page_title="户外极端环境人体热力学仿真实验系统",
    page_icon="❄️",
    layout="wide"
)

# 注入严谨风格的 CSS
st.markdown("""
<style>
    /* 全局字体与背景 */
    .stApp {
        background-color: #F8F9FA;
        font-family: "Times New Roman", "SimSun", serif; /* 衬线体显庄重 */
    }
    
    /* 标题样式 */
    h1, h2, h3 {
        color: #0F172A;
        font-weight: 700;
    }
    
    /* 学术卡片容器 */
    .academic-card {
        background-color: white;
        padding: 20px;
        border: 1px solid #E2E8F0;
        border-radius: 4px; /* 直角圆角，显严肃 */
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* 数据指标样式 */
    .metric-label { font-size: 14px; color: #64748B; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { font-size: 28px; font-weight: bold; color: #1E293B; }
    
    /* 警报状态颜色 */
    .status-normal { color: #15803D; font-weight: bold; }
    .status-warning { color: #B45309; font-weight: bold; }
    .status-danger { color: #B91C1C; font-weight: bold; }

    /* 调整侧边栏 */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心数学模型封装 (Physics Engine)
# ==========================================
def run_simulation(env_temp, wind_speed, met_val, clo_val, is_wet, duration_mins=120):
    """
    输入：环境参数
    输出：120分钟内的体温变化列表 (List), 最终状态 (Dict)
    """
    # 1. 风寒计算
    if wind_speed < 5:
        wch = env_temp
    else:
        wch = 13.12 + 0.6215 * env_temp - 11.37 * (wind_speed ** 0.16) + 0.3965 * env_temp * (wind_speed ** 0.16)
    
    # 2. 热阻修正
    real_clo = clo_val * 0.35 if is_wet else clo_val
    r_total = real_clo * 0.155 + 0.1 / (1 + 0.5 * (wind_speed / 10))
    
    # 3. 迭代计算体温
    temps = []
    curr_t = 37.0
    body_mass = 70.0
    cp = 3470.0
    area = 1.8
    heat_production = met_val * 58.15 # W/m2

    for _ in range(duration_mins + 1):
        # 散热计算
        heat_loss = (curr_t - wch) / r_total
        # 净热流
        net_flow = heat_production - heat_loss
        # 温变
        dt = (net_flow * area * 60) / (body_mass * cp)
        
        curr_t += dt
        # 物理限制
        if curr_t < env_temp: curr_t = env_temp
        
        temps.append(curr_t)
        
    return temps, wch, real_clo

# ==========================================
# 3. 辅助函数：生成动态 SVG 人体
# ==========================================
def get_human_svg(temp_val, label, clo_desc):
    """
    根据体温生成不同颜色的 SVG 人体轮廓
    """
    # 颜色映射逻辑：37度红 -> 35度蓝 -> 30度黑紫
    if temp_val >= 36.5:
        fill_color = "#E11D48" # 红色 (正常)
    elif temp_val >= 35.0:
        fill_color = "#2563EB" # 蓝色 (冷应激)
    elif temp_val >= 32.0:
        fill_color = "#4F46E5" # 深蓝 (轻度失温)
    else:
        fill_color = "#1E1B4B" # 黑紫 (重度失温)

    svg_code = f"""
    <div style="text-align: center; margin-bottom: 20px;">
        <svg viewBox="0 0 100 200" width="120" height="240">
            <!-- 简单的人体轮廓路径 -->
            <path d="M50,10 C60,10 65,18 65,28 C65,38 60,45 50,45 C40,45 35,38 35,28 C35,18 40,10 50,10 Z 
                     M30,50 L70,50 L75,100 L85,90 L95,100 L80,140 L80,140 L65,110 L65,190 L55,190 L55,140 L45,140 L45,190 L35,190 L35,110 L20,140 L5,100 L15,90 L25,100 L30,50 Z" 
                  fill="{fill_color}" stroke="#334155" stroke-width="2"/>
        </svg>
        <div style="margin-top: 10px; font-weight: bold; color: #334155;">{label}</div>
        <div style="font-size: 12px; color: #64748B;">{clo_desc}</div>
        <div style="font-size: 20px; font-weight: 700; color: {fill_color}; margin-top:5px;">{temp_val:.1f} °C</div>
    </div>
    """
    return svg_code

# ==========================================
# 4. 侧边栏：环境控制 (Input)
# ==========================================
st.sidebar.title("🔬 实验条件设定")
st.sidebar.markdown("---")

st.sidebar.subheader("1. 环境物理场 (Environment)")
env_temp = st.sidebar.slider("环境温度 / Ambient Temp (°C)", -50, 10, -20)
wind_speed = st.sidebar.slider("风速 / Wind Speed (km/h)", 0, 100, 25)

st.sidebar.subheader("2. 行为状态 (Activity)")
met_option = st.sidebar.selectbox(
    "运动代谢率 / Metabolic Rate",
    [1.0, 3.0, 6.0, 8.0],
    format_func=lambda x: f"{x} METs - " + {1.0:"静止/受伤", 3.0:"慢走", 6.0:"快速徒步", 8.0:"高强度攀登"}[x]
)

st.sidebar.subheader("3. 危险变量 (Risk Factor)")
is_wet = st.sidebar.checkbox("模拟衣物湿透 (Wet Clothing)", value=False, help="模拟汗湿或落水情况，热阻将衰减65%")

st.sidebar.markdown("---")
st.sidebar.info("本模型基于 *Osczevski-Bluestein* 风寒指数模型与人体热平衡方程构建。\n\n适用于《户外运动安全》课程教学演示。")

# ==========================================
# 5. 主界面：对比实验区
# ==========================================

st.title("🏔️ 户外运动失温伤害虚拟仿真实验")
st.markdown("**实验目的：** 研究在同一极端环境下，不同着装方案（热阻 Clo）对人体核心体温维持能力的差异性分析。")

# 计算三个对照组
# 组1：轻装 (T恤/薄外套) - 0.5 Clo
temps_1, wch, _ = run_simulation(env_temp, wind_speed, met_option, 0.5, is_wet)
# 组2：标准 (冲锋衣套装) - 1.5 Clo
temps_2, _, _ = run_simulation(env_temp, wind_speed, met_option, 1.5, is_wet)
# 组3：专业 (高山连体羽绒) - 3.5 Clo
temps_3, _, _ = run_simulation(env_temp, wind_speed, met_option, 3.5, is_wet)

# --- 模块一：实时状态对比 (Virtual Avatars) ---
st.markdown("### 1. 120分钟后人体热力学状态模拟")
st.markdown("通过数值模拟生成的三组虚拟人体模型，颜色代表核心体温分布（红=正常，蓝=失温）。")

with st.container():
    # 使用列布局显示三个“人”
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("<div class='academic-card'>", unsafe_allow_html=True)
        st.markdown(get_human_svg(temps_1[-1], "实验组 A", "轻薄衣物 (0.5 Clo)"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown("<div class='academic-card'>", unsafe_allow_html=True)
        st.markdown(get_human_svg(temps_2[-1], "实验组 B", "标准户外装 (1.5 Clo)"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("<div class='academic-card'>", unsafe_allow_html=True)
        st.markdown(get_human_svg(temps_3[-1], "实验组 C", "专业高山向导 (3.5 Clo)"), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- 模块二：数据可视化 (Academic Chart) ---
st.markdown("### 2. 核心体温时变曲线 (Time-Temperature Analysis)")

# 构建 Plotly 图表
fig = go.Figure()

# 绘制三条曲线
fig.add_trace(go.Scatter(
    x=np.arange(121), y=temps_1, 
    mode='lines', name='组A: 0.5 Clo',
    line=dict(color='#EF4444', width=2, dash='dash') # 红色虚线，表示危险
))
fig.add_trace(go.Scatter(
    x=np.arange(121), y=temps_2, 
    mode='lines', name='组B: 1.5 Clo',
    line=dict(color='#F59E0B', width=3) # 橙色
))
fig.add_trace(go.Scatter(
    x=np.arange(121), y=temps_3, 
    mode='lines', name='组C: 3.5 Clo',
    line=dict(color='#10B981', width=3) # 绿色
))

# 绘制安全阈值区域
fig.add_hrect(y0=35, y1=38, fillcolor="green", opacity=0.05, line_width=0, annotation_text="安全区", annotation_position="top left")
fig.add_hrect(y0=32, y1=35, fillcolor="orange", opacity=0.05, line_width=0, annotation_text="轻度失温区", annotation_position="top left")
fig.add_hrect(y0=20, y1=32, fillcolor="red", opacity=0.05, line_width=0, annotation_text="重度失温区", annotation_position="top left")

# 设置严格的学术风格布局
fig.update_layout(
    title=dict(text=f'环境温度 {env_temp}°C / 风速 {wind_speed} km/h 条件下的体温演变', font=dict(size=16)),
    xaxis=dict(
        title='暴露时长 (Exposure Time) [min]', # 明确的X轴标签
        showgrid=True,
        gridcolor='#E2E8F0',
        zeroline=True,
    ),
    yaxis=dict(
        title='核心体温 (Core Temp) [°C]', # 明确的Y轴标签
        showgrid=True,
        gridcolor='#E2E8F0',
        range=[min(28, min(temps_1)-1), 38]
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    plot_bgcolor='white', # 白底，学术规范
    height=500,
    margin=dict(l=60, r=40, t=80, b=60) # 增加边距，防止标签被切
)

st.plotly_chart(fig, use_container_width=True)

# --- 模块三：结论与分析 ---
st.markdown("### 3. 实验数据与分析结论")
result_col1, result_col2 = st.columns([1, 2])

with result_col1:
    st.markdown("""
    <div class='academic-card'>
        <div class='metric-label'>当前体感温度 (Wind Chill)</div>
        <div class='metric-value' style='color:#3B82F6'>%.1f °C</div>
    </div>
    """ % wch, unsafe_allow_html=True)

with result_col2:
    # 动态生成结论
    conclusion = ""
    if temps_3[-1] > 36.0:
        conclusion += "✅ **专业装备有效性验证：** 在当前环境下，高热阻装备（3.5 Clo）能有效维持体温平衡。<br>"
    if temps_2[-1] < 35.0:
        conclusion += "⚠️ **常规装备局限性：** 普通户外装（1.5 Clo）不足以应对该极端环境，需在60分钟内寻找避难所。<br>"
    if temps_1[-1] < 32.0:
        conclusion += "☠️ **失温风险预警：** 轻装组在当前风寒条件下将迅速进入重度失温状态，有生命危险。"
    
    if is_wet:
        conclusion += "<br><br><strong>💧 潮湿效应显著：</strong> 实验数据显示，潮湿导致衣物热阻效能降低约 65%，加速了热量流失。"

    st.info(conclusion)
