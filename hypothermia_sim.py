import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ==========================================
# 1. 页面基础设置
# ==========================================
st.set_page_config(
    page_title="Fiala人体热调节模型 - 教学仿真系统",
    page_icon="🌡️",
    layout="wide"
)

# 注入 CSS 修复图形渲染问题
st.markdown("""
<style>
    /* 强制 SVG 容器居中并显示边框 */
    .svg-container {
        display: flex;
        justify-content: center;
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    
    /* 数据表格样式 */
    .dataframe { font-size: 14px !important; }
    
    /* 标题样式 */
    h1, h2, h3 { font-family: 'Times New Roman', serif; color: #1e293b; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. 核心：Fiala 多节段生物热算法 (Physics Engine)
# ==========================================
class BodySegment:
    def __init__(self, name_en, name_cn, mass, area, basal_met, vaso_factor):
        self.name_en = name_en   # 英文ID (用于代码索引)
        self.name_cn = name_cn   # 中文名 (用于显示)
        self.mass = mass         # 质量 kg
        self.area = area         # 表面积 m2
        self.temp_skin = 33.0    # 初始皮温
        self.temp_core = 37.0    # 初始核温
        self.basal_met = basal_met # 基础代谢 W
        self.vaso_factor = vaso_factor # 血管收缩敏感度 (关键参数)
        self.history = [33.0]    # 历史记录

def run_simulation(env_temp, wind_speed, clo, met, is_wet, duration=120):
    # --- A. 初始化人体模型 (基于 Fiala/Berkeley 参数) ---
    segments = {
        "Head":  BodySegment("Head", "头部", 4.5, 0.14, 12.0, 0.1),
        "Trunk": BodySegment("Trunk", "躯干", 30.0, 0.55, 45.0, 0.1),
        "Arms":  BodySegment("Arms", "手臂", 4.0, 0.26, 3.0, 0.8), # 敏感度中
        "Hands": BodySegment("Hands", "手部", 0.4, 0.08, 0.5, 3.0), # 敏感度极高
        "Legs":  BodySegment("Legs", "腿部", 12.0, 0.60, 8.0, 0.8), # 敏感度中
        "Feet":  BodySegment("Feet", "脚部", 1.0, 0.14, 0.5, 3.0)  # 敏感度极高
    }
    
    # 模拟循环
    central_blood_temp = 37.0 # 核心血温
    time_points = np.arange(0, duration + 1)
    
    # 风效修正 (Osczevski)
    v_eff = wind_speed if wind_speed < 5 else wind_speed * 0.6
    
    for t in range(duration):
        total_blood_return_heat = 0
        total_met_heat = 0
        
        # 1. 遍历计算每个部位
        for key, seg in segments.items():
            # A. 产热 (Metabolism)
            # 运动时，大肌群(腿/躯干)产热多，手脚产热少
            act_factor = met
            if key in ["Hands", "Feet", "Head"]: act_factor = 1.0 + (met-1)*0.1
            
            q_met = seg.basal_met * act_factor
            total_met_heat += q_met
            
            # B. 散热 (Heat Loss)
            # 潮湿修正：热阻衰减
            real_clo = clo * 0.35 if is_wet else clo
            # 头部和手部通常覆盖较少，做修正
            if key in ["Head", "Hands"]: local_clo = real_clo * 0.3
            else: local_clo = real_clo
            
            r_total = 0.155 * local_clo + 0.1 / (1 + 0.5*v_eff)
            q_loss = seg.area * (seg.temp_skin - env_temp) / r_total
            
            # C. 血液灌注 (Blood Perfusion - 论文核心)
            # 血管收缩逻辑：核心越冷，末端供血越少
            vaso_response = 1.0
            if central_blood_temp < 36.8:
                delta = 36.8 - central_blood_temp
                # 敏感度越高(手脚)，血流关闭得越快
                vaso_response = 1.0 / (1.0 + seg.vaso_factor * delta * 8.0)
            
            # 血液带来的热量 (从核心带给皮肤)
            q_blood = 18.0 * seg.mass * vaso_response * (central_blood_temp - seg.temp_skin) / 60
            
            # 记录回心血流的热损失效应 (用于冷却核心)
            total_blood_return_heat -= q_blood 
            
            # D. 温度更新 (热容模型)
            net_heat_joules = (q_met + q_blood - q_loss) * 60 # 1分钟
            dt = net_heat_joules / (seg.mass * 3470)
            
            seg.temp_skin += dt
            if seg.temp_skin < env_temp: seg.temp_skin = env_temp # 物理极值
            
            seg.history.append(seg.temp_skin)
            
        # 2. 更新核心血温 (简化版核心热平衡)
        # 核心受代谢加热，受回心冷血冷却
        core_mass = 50.0 # 核心质量
        core_dt = (total_met_heat * 1.5 + total_blood_return_heat) * 60 / (core_mass * 3470)
        central_blood_temp += core_dt
        
        # 生理稳态微调 (模拟寒战勉强维持)
        if central_blood_temp < 37.0: central_blood_temp += 0.002 
            
    return segments, time_points, central_blood_temp

# ==========================================
# 3. 可视化：解剖级 SVG 生成器 (Visual Engine)
# ==========================================
def render_human_svg(segments):
    # 颜色映射 (蓝 -> 红)
    def get_color(t):
        if t < 10: return "#09090b" # 冻结 (黑)
        if t < 20: return "#172554" # 极寒 (深蓝)
        if t < 28: return "#2563eb" # 失温 (蓝)
        if t < 33: return "#f59e0b" # 冷 (橙)
        return "#dc2626" # 暖 (红)

    cols = {k: get_color(v.history[-1]) for k, v in segments.items()}
    temps = {k: v.history[-1] for k, v in segments.items()}

    # SVG 绘图代码 (显式指定了 width/height 防止塌陷)
    svg = f"""
    <svg width="300" height="550" viewBox="0 0 300 550" xmlns="http://www.w3.org/2000/svg">
        <!-- 头部 -->
        <g id="head">
            <path d="M130,50 Q130,20 150,20 Q170,20 170,50 Q170,70 150,70 Q130,70 130,50 Z" 
                  fill="{cols['Head']}" stroke="#333" stroke-width="2"/>
            <text x="190" y="55" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Head']:.1f}°C</text>
            <line x1="170" y1="50" x2="185" y2="50" stroke="#666" stroke-width="1"/>
        </g>
        
        <!-- 躯干 -->
        <g id="trunk">
            <path d="M120,70 L180,70 L190,200 L110,200 Z" 
                  fill="{cols['Trunk']}" stroke="#333" stroke-width="2"/>
            <text x="150" y="140" text-anchor="middle" font-family="Arial" font-size="14" fill="white" font-weight="bold">{temps['Trunk']:.1f}</text>
        </g>
        
        <!-- 手臂 (左/右) -->
        <g id="arms">
            <path d="M120,70 L90,160 L110,170 L130,80 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
            <path d="M180,70 L210,160 L190,170 L170,80 Z" fill="{cols['Arms']}" stroke="#333" stroke-width="2"/>
        </g>
        
        <!-- 手部 (重点) -->
        <g id="hands">
            <path d="M90,160 L80,190 L100,200 L110,170 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
            <path d="M210,160 L220,190 L200,200 L190,170 Z" fill="{cols['Hands']}" stroke="#333" stroke-width="2"/>
            
            <!-- 标签 -->
            <line x1="80" y1="190" x2="50" y2="190" stroke="#666" stroke-width="1"/>
            <text x="10" y="195" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Hands']:.1f}°C</text>
        </g>
        
        <!-- 腿部 -->
        <g id="legs">
            <path d="M110,200 L100,400 L140,400 L145,200 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
            <path d="M190,200 L200,400 L160,400 L155,200 Z" fill="{cols['Legs']}" stroke="#333" stroke-width="2"/>
        </g>
        
        <!-- 脚部 (重点) -->
        <g id="feet">
            <path d="M100,400 L90,430 L130,430 L140,400 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
            <path d="M200,400 L210,430 L170,430 L160,400 Z" fill="{cols['Feet']}" stroke="#333" stroke-width="2"/>
            
            <!-- 标签 -->
            <line x1="210" y1="430" x2="240" y2="430" stroke="#666" stroke-width="1"/>
            <text x="245" y="435" font-family="Arial" font-size="14" fill="#333" font-weight="bold">{temps['Feet']:.1f}°C</text>
        </g>
        
        <!-- 图例 -->
        <rect x="50" y="480" width="200" height="10" fill="url(#grad1)" />
        <defs>
            <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#09090b;stop-opacity:1" />
                <stop offset="50%" style="stop-color:#2563eb;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#dc2626;stop-opacity:1" />
            </linearGradient>
        </defs>
        <text x="50" y="510" font-size="12">冻结 (0°C)</text>
        <text x="250" y="510" font-size="12" text-anchor="end">正常 (37°C)</text>
    </svg>
    """
    return svg

# ==========================================
# 4. 主程序 (Main App)
# ==========================================

# --- 侧边栏 ---
st.sidebar.title("🎮 实验控制台")
st.sidebar.markdown("---")
env_temp = st.sidebar.slider("环境温度 (°C)", -40, 10, -10)
wind_speed = st.sidebar.slider("风速 (km/h)", 0, 80, 20)
met_val = st.sidebar.selectbox("运动状态", [1.0, 3.0, 6.0, 8.0], format_func=lambda x: f"{x} METs")
clo_val = st.sidebar.slider("服装热阻 (Clo)", 0.5, 4.0, 1.5)
is_wet = st.sidebar.checkbox("衣物湿透 (Danger)", False)

# --- 运行计算 ---
segments, time_x, core_temp = run_simulation(env_temp, wind_speed, clo_val, met_val, is_wet)

# --- 主界面 ---
st.title("🏔️ 户外运动失温伤害虚拟仿真系统 (Ver 4.1)")
st.markdown("本系统模拟人体在极端环境下的热调节机制，重点展示 **“核心-外周”温差** 与 **逆流热交换** 现象。")

col1, col2 = st.columns([1, 1.5])

# --- 左侧：人体可视化 ---
with col1:
    st.subheader("1. 实时热成像 (Simulation)")
    # 使用 div 容器包裹，确保样式生效
    st.markdown(f'<div class="svg-container">{render_human_svg(segments)}</div>', unsafe_allow_html=True)
    
    # 核心体温警报
    alert_color = "green"
    status_text = "核心体温正常"
    if core_temp < 35: 
        alert_color = "red"
        status_text = "警告：进入失温状态！"
    elif core_temp < 36.5:
        alert_color = "orange"
        status_text = "注意：冷应激反应"
        
    st.markdown(f"""
    <div style="background-color:{alert_color}; padding:10px; border-radius:5px; color:white; text-align:center;">
        <h3>{status_text}</h3>
        <p>核心血温: {core_temp:.1f} °C</p>
    </div>
    """, unsafe_allow_html=True)

# --- 右侧：数据与图表 ---
with col2:
    st.subheader("2. 生理参数监测 (Monitoring)")
    
    # A. 交互式图表
    fig = go.Figure()
    
    # 重点画手部和躯干
    fig.add_trace(go.Scatter(x=time_x, y=segments['Trunk'].history, name="躯干 (Core)", line=dict(color="#f97316", width=3)))
    fig.add_trace(go.Scatter(x=time_x, y=segments['Hands'].history, name="手部 (Extremity)", line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=time_x, y=segments['Feet'].history, name="脚部 (Extremity)", line=dict(color="#1e3a8a", width=3)))
    
    fig.update_layout(
        title="躯干 vs 四肢末端 温度分离现象",
        xaxis_title="暴露时间 (分钟)",
        yaxis_title="皮肤温度 (°C)",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # B. 数据概览表 (数据备份，防止图形看不清)
    st.subheader("3. 实时数值面板 (Data Panel)")
    
    # 构造数据表
    data = []
    for k, v in segments.items():
        start_t = v.history[0]
        end_t = v.history[-1]
        drop = start_t - end_t
        data.append({
            "部位": v.name_cn,
            "初始温度": f"{start_t:.1f}°C",
            "当前温度": f"{end_t:.1f}°C",
            "温降幅度": f"{drop:.1f}°C",
            "状态": "❄️ 冻伤风险" if end_t < 15 else ("🔵 失温" if end_t < 30 else "✅ 正常")
        })
    
    df = pd.DataFrame(data)
    st.dataframe(df, hide_index=True, use_container_width=True)

# --- 底部：原理说明 ---
st.markdown("---")
st.info("""
**教学原理说明：**
当您增加风速或降低气温时，请注意观察 **“躯干”** 与 **“手/脚”** 的温差。
模型复现了 *Fiala et al.* 的 **血管收缩机制 (Vasoconstriction)**：人体为了保全核心器官（心脑肺）的温度，
会主动切断流向四肢的血液。因此，您会看到手脚温度迅速下降（变蓝/黑），而躯干温度下降较慢。
""")
