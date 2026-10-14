import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# ===================== 农业种植决策智能体【修复产量换算】 =====================
class CropDecisionAgent:
    def __init__(self):
        self.data = self.build_dataset()
        self.feature_cols = ["crop_code", "area_harvested"]
        self.target_yield = "yield"
        self.rf_yield, self.Xy_train, self.Xy_test, self.yy_train, self.yy_test = self.train_yield_model()
        self.crop_name_map = {0:"玉米(Maize)", 1:"水稻(Rice)"}

    def build_dataset(self):
        # 读取FAOSTAT原始csv
        df_raw = pd.read_csv("faostat_qcl.csv")
        # 筛选中国区域
        df_china = df_raw[df_raw["Area"] == "China"].copy()
        # 筛选收获面积、单产两类指标
        df_filter = df_china[df_china["Element"].isin(["Area harvested", "Yield"])].copy()
        # 透视表，把同一作物同年份的面积与产量合并到一行
        df_pivot = df_filter.pivot_table(
            index=["Item", "Year"],
            columns="Element",
            values="Value",
            aggfunc="mean"
        ).reset_index()
        df_pivot.columns = ["Item", "Year", "area_harvested", "yield"]
        df_pivot = df_pivot.dropna()
        # 作物编码映射
        def encode_crop(name):
            if "Maize" in str(name):
                return 0
            elif "Rice" in str(name):
                return 1
            else:
                return np.nan
        df_pivot["crop_code"] = df_pivot["Item"].apply(encode_crop)
        df_pivot = df_pivot.dropna(subset=["crop_code"])
        df_pivot["crop_code"] = df_pivot["crop_code"].astype(int)
        return df_pivot

    def train_yield_model(self):
        data = self.data
        X = data[self.feature_cols]
        y = data[self.target_yield]
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        return model, X_train, X_test, y_train, y_test

    def predict_single(self, crop_code, area_mu, seed_cost, fer_cost, pest_cost, labor_cost, land_cost, drought, pest, price):
        # 亩转公顷：1公顷 =15亩
        area_ha = area_mu / 15
        # FAOSTAT模型输出基础单产（hg/ha，百克/公顷），*10修正量级
        input_data = pd.DataFrame({
            "crop_code":[crop_code],
            "area_harvested":[area_ha]
        })
        base_yield_hg_ha = self.rf_yield.predict(input_data)[0] * 10
        # 农学经验减产系数（仅交互演示，网页会标注说明）
        drought_factor = 1 - drought * 0.08
        pest_factor = 1 - pest * 0.07
        final_yield_hg_ha = base_yield_hg_ha * drought_factor * pest_factor
        # 单位换算：hg/ha → kg/亩
        # 1 hg = 0.1 kg；1公顷=15亩
        kg_per_mu = final_yield_hg_ha * 0.1 / 15
        total_production_kg = kg_per_mu * area_mu
        # 成本：这里各项成本为【每亩成本】
        total_cost = (seed_cost + fer_cost + pest_cost + labor_cost + land_cost) * area_mu
        final_profit = total_production_kg * price - total_cost
        return round(kg_per_mu,2), round(final_profit,2)

    def generate_candidate_plans(self, drought, pest, seed_cost, fer_cost, pest_cost, labor_cost, land_cost, price):
        candidate_list = []
        crop_list = [0,1]
        area_mu_list = np.linspace(100, 10000, 10)
        for crop in crop_list:
            for area_mu in area_mu_list:
                pred_yield, pred_profit = self.predict_single(crop, area_mu, seed_cost, fer_cost, pest_cost, labor_cost, land_cost, drought, pest, price)
                candidate_list.append({
                    "crop_code": crop,
                    "作物名称": self.crop_name_map[crop],
                    "种植面积(亩)": round(area_mu,1),
                    "预测单产(kg/亩)": pred_yield,
                    "预测净收益(元)": pred_profit
                })
        return candidate_list

    def get_suggestion(self, current_crop, current_area_mu, drought, pest, pred_yield, pred_profit, candidates):
        df_cand = pd.DataFrame(candidates)
        best_yield_plan = df_cand.loc[df_cand["预测单产(kg/亩)"].idxmax()]
        best_profit_plan = df_cand.loc[df_cand["预测净收益(元)"].idxmax()]
        avg_yield = df_cand["预测单产(kg/亩)"].mean()
        avg_profit = df_cand["预测净收益(元)"].mean()

        advice_text = []
        advice_text.append("# 🤖 智能体决策分析报告")
        advice_text.append(f"> 当前方案：作物【{self.crop_name_map[current_crop]}】，种植面积 {current_area_mu} 亩")
        advice_text.append(f"> 环境条件：干旱等级{drought}，虫害等级{pest}")
        advice_text.append(f"> 同条件批量模拟方案平均单产 {round(avg_yield,2)} kg/亩，平均净收益 {round(avg_profit,2)} 元")

        if pred_yield >= avg_yield and pred_profit >= avg_profit:
            advice_text.append("\n✅ 智能评估：该方案综合表现优于平均水平，具备较好可行性。")
        elif pred_yield >= avg_yield and pred_profit < avg_profit:
            advice_text.append("\n⚠️ 智能评估：作物单产潜力尚可，但整体收益低于平均，建议优化投入成本。")
        elif pred_yield < avg_yield and pred_profit >= avg_profit:
            advice_text.append("\n⚠️ 智能评估：收益达标，但单产存在上限，建议关注田间防灾减灾。")
        else:
            advice_text.append("\n❌ 智能评估：产量与收益均低于平均基准，方案风险偏高，建议重新调整种植规划。")

        advice_text.append("\n## 🎯 推荐备选方案")
        advice_text.append(f"【产量优先方案】作物：{best_yield_plan['作物名称']}，面积：{best_yield_plan['种植面积(亩)']}亩，预测单产：{best_yield_plan['预测单产(kg/亩)']}kg/亩，净收益：{best_yield_plan['预测净收益(元)']}元")
        advice_text.append(f"【收益优先方案】作物：{best_profit_plan['作物名称']}，面积：{best_profit_plan['种植面积(亩)']}亩，预测单产：{best_profit_plan['预测单产(kg/亩)']}kg/亩，净收益：{best_profit_plan['预测净收益(元)']}元")

        advice_text.append("\n## 💡 智能决策提示")
        if drought >= 0.6:
            advice_text.append("👉 当前干旱等级较高，建议配套灌溉设施，降低减产风险。")
        if pest >=0.6:
            advice_text.append("👉 当前虫害风险较高，建议提前做好病虫害预防工作。")
        if pred_profit < best_profit_plan["预测净收益(元)"] * 0.8:
            advice_text.append("👉 当前方案收益提升空间较大，可以参考收益优先方案调整种植规模。")
        advice_text.append("\n> 提示：产量基线来自FAOSTAT多年汇总真实农业数据；干旱、虫害因子为农学经验模拟，仅用于场景演示。")
        return "\n".join(advice_text), best_yield_plan, best_profit_plan

# -------------------------- Streamlit页面配置 --------------------------
st.set_page_config(page_title="农业种植决策智能体", layout="wide")
st.title("🌾 农业种植决策智能体")
st.markdown("""
> **模型说明：**基础作物单产由FAOSTAT联合国粮农组织多年汇总农业统计数据，以作物类型、种植面积为特征训练随机森林模型预测；干旱、病虫害为农学经验模拟因子，用于场景交互演示，仅供种植决策参考。
> 单位说明：面积单位为亩；各项成本为每亩投入成本；单产单位kg/亩；收益单位元。
""")
st.divider()

@st.cache_resource
def load_agent():
    agent = CropDecisionAgent()
    return agent

agent = load_agent()

col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("📝 参数输入面板")
    crop_dict = {0:"玉米(Maize)", 1:"水稻(Rice)"}
    crop_code = st.selectbox("选择作物", options=[0,1], format_func=lambda x:crop_dict[x])
    area_mu = st.slider("收获面积（亩）", min_value=100.0, max_value=10000.0, value=1000.0, step=100.0)

    st.subheader("💰 每亩投入成本")
    seed_cost = st.slider("种子成本（元/亩）", min_value=20.0, max_value=200.0, value=60.0, step=5.0)
    fer_cost = st.slider("肥料成本（元/亩）", min_value=50.0, max_value=400.0, value=120.0, step=10.0)
    pest_cost = st.slider("防虫农药成本（元/亩）", min_value=10.0, max_value=150.0, value=40.0, step=5.0)
    labor_cost = st.slider("人工成本（元/亩）", min_value=80.0, max_value=500.0, value=160.0, step=10.0)
    land_cost = st.slider("土地成本（元/亩）", min_value=60.0, max_value=400.0, value=100.0, step=10.0)

    st.subheader("🌦️ 环境风险参数")
    drought = st.slider("干旱等级（0无干旱 ~1重度干旱）", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
    pest = st.slider("虫害等级（0无虫害 ~1重度虫害）", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

    st.subheader("📦 农产品售价")
    price = st.slider("作物售价（元/kg）", min_value=1.0, max_value=4.0, value=2.2, step=0.1)

    run_btn = st.button("🚀 智能体启动预测", type="primary")

with col_right:
    st.header("📊 预测结果与智能分析")
    if run_btn:
        pred_yield, pred_profit = agent.predict_single(crop_code, area_mu, seed_cost, fer_cost, pest_cost, labor_cost, land_cost, drought, pest, price)
        col_res1, col_res2 = st.columns([1,1])
        with col_res1:
            st.metric(label="预测单产(kg/亩)", value=f"{pred_yield}")
        with col_res2:
            st.metric(label="预测净收益(元)", value=f"{pred_profit}")

        candidates = agent.generate_candidate_plans(drought=drought, pest=pest, seed_cost=seed_cost, fer_cost=fer_cost, pest_cost=pest_cost, labor_cost=labor_cost, land_cost=land_cost, price=price)
        suggestion_text, best_yield, best_profit = agent.get_suggestion(crop_code, area_mu, drought, pest, pred_yield, pred_profit, candidates)
        st.markdown(suggestion_text)

st.divider()
st.header("📋 智能体批量生成候选方案库")
if run_btn:
    st.dataframe(pd.DataFrame(candidates), use_container_width=True)