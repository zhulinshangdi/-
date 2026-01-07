import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 页面配置与样式
# ==========================================
st.set_page_config(
    page_title="户外运动失温仿真教学系统",
    page_icon="🏔️",
    layout="wide"
)

# 自定义一些CSS让界面更像教学软件
st.markdown("""
<style>
    .big-font { font-size:20px !important; color: #333; }
    .highlight { background-color: #f0f2f6; padding: 10px; border-radius: 10px; }
    .danger { color: #ff4b4b; font-weight: bold; }
    .safe { color: #09ab3b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏔️ 珠峰攀登环境：人体体温调节与失温仿真模型")
st.markdown("本系统用于模拟**户外极端环境**下，不同**气象条件**、**运动强度**及**着装方案**对人体核心体温的影响。")

# ==========================================
# 2. 侧边栏：交互控制台 (输入层)
# ==========================================
st.sidebar.header("⚙️ 实验参数设置")

# --- A. 环境设置 ---
st.sidebar.subheader("1. 环境物理场")
env_temp = st.sidebar.slider("环境温度 (°C)", -50, 10, -20, help="珠峰顶端常年在-30°C左右")
wind_speed = st.sidebar.slider("风速 (km/h)", 0, 100, 30, help="风速越大，风寒效应越明显")

# --- B. 运动行为 ---
st.sidebar.subheader("2. 运动状态")
activity_level = st.sidebar.selectbox(
    "当前动作",
    options=["静止/受伤等待", "轻度活动 (慢走)", "中度活动 (徒步)", "高强度 (攀冰/冲顶)"],
    index=2
)
# 将选项映射为 METs (代谢当量)
met_map = {
    "静止/受伤等待": 1.0,
    "轻度活动 (慢走)": 2.5,
    "中度活动 (徒步)": 4.5,
    "高强度 (攀冰/冲顶)": 8.0
}
mets = met_map[activity_level]

# --- C. 装备系统 ---
st.sidebar.subheader("3. 服装与装备")
clothing_type = st.sidebar.selectbox(
    "穿着方案",
    options=["单薄衣物 (0.5 Clo)", "常规冲锋衣套装 (1.5 Clo)", "专业高山羽绒连体服 (3.5 Clo)"],
    index=1
)
clo_map = {
    "单薄衣物 (0.5 Clo)": 0.5,
    "常规冲锋衣套装 (1.5 Clo)": 1.5,
    "专业高山羽绒连体服 (3.5 Clo)": 3.5
}
base_clo = clo_map[clothing_type]

# 核心交互变量：潮湿
is_wet = st.sidebar.checkbox("⚠️ 警告：内层衣物是否湿透？", value=False, help="汗湿或雪水浸湿会严重降低保温能力")

# ==========================================
# 3. 模型计算核心 (逻辑层)
# ==========================================

def calculate_simulation(t_env, wind, met_val, clo_val, wet_status):
    # 1. 计算风寒温度 (Osczevski-Bluestein公式)
    # 这是一个气象学公式，计算"感觉有多冷"
    if wind < 5:
        wind_chill = t_env
    else:
        # v 需要转换为 m/s 用于部分计算，这里风寒公式用 km/h 适配
        wind_chill = 13.12 + 0.6215 * t_env - 11.37 * (wind ** 0.16) + 0.3965 * t_env * (wind ** 0.16)
    
    # 2. 修正服装热阻
    # 如果湿透，棉/羽绒热阻仅剩 30%-40%
    real_clo = clo_val * 0.35 if wet_status else clo_val
    # 转换为标准热阻单位 (m2·K/W)
    r_clothing = real_clo * 0.155 
    r_air = 0.1 / (1 + 0.5 * (wind / 10)) # 风越大，空气层热阻越小
    r_total = r_clothing + r_air

    # 3. 产热 (W/m2)
    heat_production = met_val * 58.15 
    
    # 4. 散热 (W/m2)
    # 简化物理模型：热流 = 温差 / 热阻
    # 假设核心体温初始 37度
    heat_loss = (37.0 - wind_chill) / r_total
    
    # 5. 净热量平衡
    net_heat = heat_production - heat_loss
    
    return wind_chill, net_heat, real_clo

# 运行单次计算用于仪表盘
wc, net_q, actual_clo = calculate_simulation(env_temp, wind_speed, mets, base_clo, is_wet)

# ==========================================
# 4. 可视化输出 (UI层)
# ==========================================

# --- 顶部仪表盘 ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("环境温度", f"{env_temp} °C")
col2.metric("体感温度 (风寒)", f"{wc:.1f} °C", delta=f"{wc - env_temp:.1f} °C", delta_color="inverse")
col3.metric("实际保暖值 (Clo)", f"{actual_clo:.2f}", delta="-65%" if is_wet else "正常", delta_color="inverse")

# 判断热平衡状态
status_text = ""
status_color = ""
if net_q > 0:
    status_text = "体温维持/上升 (安全)"
    status_color = "safe"
else:
    status_text = "⚠️ 体温正在流失 (危险)"
    status_color = "danger"

col4.markdown(f"#### 状态: <span class='{status_color}'>{status_text}</span>", unsafe_allow_html=True)


# --- 核心：动态时序模拟 ---
st.markdown("---")
st.subheader("📉 核心体温变化预测 (未来2小时)")

# 模拟算法：基于简单热容量模型
# Q = cmΔT -> ΔT = Q / cm
simulation_minutes = 120
time_x = np.arange(0, simulation_minutes)
temp_y = []
current_core_temp = 37.0
body_mass = 70 # kg
specific_heat = 3470 # J/(kg·C)
surface_area = 1.8 # m2

# 记录失温阶段
hypothermia_onset = None # 开始失温时间

for t in time_x:
    # 每一分钟计算一次新的体温
    # 这里的 net_q 是 W/m2 (焦耳/秒/平方米)
    # 每分钟总热量变化 (Joules) = net_q * Area * 60s
    total_joules_change = net_q * surface_area * 60
    
    # 温度变化量
    dt = total_joules_change / (body_mass * specific_heat)
    
    # 加上生理调节反馈（简化版）：
    # 如果体温降低，会寒战(Shivering)，产热增加，但这里为了教学展示"如果不干预会怎样"，暂不加寒战补偿，
    # 这样更能体现物理环境的残酷性。
    
    current_core_temp += dt
    
    # 物理限制：尸体温度不会低于环境温度
    if current_core_temp < env_temp:
        current_core_temp = env_temp
        
    temp_y.append(current_core_temp)
    
    # 记录第一次跌破35度的时间
    if current_core_temp < 35.0 and hypothermia_onset is None:
        hypothermia_onset = t

# 绘图
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(time_x, temp_y, color='#ff4b4b', linewidth=2, label='核心体温')

# 绘制安全警戒线
ax.axhline(y=35, color='blue', linestyle='--', alpha=0.5, label='轻度失温界限 (35°C)')
ax.axhline(y=32, color='purple', linestyle='--', alpha=0.5, label='重度失温界限 (32°C)')

ax.set_ylim(bottom=min(25, min(temp_y)-1), top=38)
ax.set_xlabel("暴露时间 (分钟)")
ax.set_ylabel("核心体温 (°C)")
ax.grid(True, alpha=0.3)
ax.legend()

st.pyplot(fig)

# --- 教学反馈区 ---
c1, c2 = st.columns([2, 1])

with c1:
    st.info("💡 **教学观察点**：尝试勾选侧边栏的 **'内层衣物湿透'**，观察体温曲线斜率的变化。你会发现潮湿对失温的加速作用比单纯的低温更可怕。")

with c2:
    if hypothermia_onset:
        st.error(f"🛑 **危险预警**\n\n以当前状态，预计 **{hypothermia_onset} 分钟** 后进入失温状态 (Core Temp < 35°C)。\n\n**建议操作：**\n1. 增加衣物\n2. 寻找避风处\n3. 更换干衣")
    else:
        st.success("✅ **安全评估**\n\n在当前环境下，2小时内体温能维持在安全范围内。")

# --- 底部：人体热力图概念演示 ---
st.markdown("---")
st.subheader("🧖‍♂️ 人体热分布 (概念可视化)")

# 根据最终温度决定显示哪张图（这里用色块模拟，实际开发可用图片）
final_temp = temp_y[-1]
color_hex = "#ff0000" # 正常红
if final_temp < 32: color_hex = "#2b0057" # 深度紫
elif final_temp < 35: color_hex = "#0066ff" # 失温蓝
elif final_temp < 36.5: color_hex = "#ffaa00" # 发冷橙

st.markdown(f"""
<div style="display:flex; justify-content:center; align-items:center; flex-direction:column;">
    <div style="width: 200px; height: 300px; background: linear-gradient(to bottom, {color_hex}, {color_hex}AA); 
                border-radius: 100px; display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; box-shadow: 0 0 20px {color_hex}; transition: all 0.5s;">
        人体核心
    </div>
    <p style="margin-top:10px; color:#666;">当前体表/核心颜色示意</p>
</div>
""", unsafe_allow_html=True)