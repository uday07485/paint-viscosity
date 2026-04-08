import streamlit as st
import pandas as pd
import joblib
from xgboost import XGBRegressor

# -------------------------------
# MUST be first command
# -------------------------------
st.set_page_config(
    page_title="Paint Viscosity Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------------------
# Load model files
# -------------------------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")


# ===============================
# SIDEBAR NAVIGATION
# ===============================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["📄 Description", "📈 Prediction"]
)

# ===============================
# 📄 DESCRIPTION PAGE
# ===============================
if page == "📄 Description":
    st.title("🧪 Paint Viscosity Prediction System")

    st.markdown("""
    ## Welcome
                
### 🔹 Overview
This project is a machine learning-based system developed to predict the viscosity of paint (in centipoise - cP) using raw material composition, process parameters, and environmental conditions.
The system helps in understanding how different factors influence paint behavior and ensures better quality control in chemical manufacturing.
             
---
             
### 🔹 Problem Statement
-In paint manufacturing, predicting the viscosity before production is difficult and time-consuming.  
-Traditional methods require trial-and-error, leading to increased cost and production delay.

👉 This system solves that problem by **predicting viscosity instantly using ML models**.

---

### 🔹 Objective
The main objective of this project is to:
1. Predict paint viscosity accurately
2. Reduce manual testing and errors
3. Improve consistency in paint production
4. Help industries choose the right formulation
5. Support faster decision-making in manufacturing
             
---
             
### 🔹 Workflow
1. User enters input values through the interface
2. Data is converted into structured format
3. Categorical variables are encoded
4. Input is aligned with training features
5. Data is scaled using StandardScaler
6. Machine learning model predicts viscosity
             
---

### 🔹 Model Used
1. XGBoost Regressor is used for viscosity prediction
2. It provides high accuracy and handles complex relationships between variables
             
---

### 🔹 Importance of Viscosity
Viscosity is a critical property in paint manufacturing because it:
1. Controls paint flow and spreadability
2. Affects coating thickness
3. Impacts drying time
4. Determines surface finish quality
5. Ensures product consistency
             
---

### 🔹 Applications
1. Paint manufacturing industries
2. Chemical production units
3. Quality control laboratories
4. Industrial coating applications
                          
---

### 🔹 Features
1. Easy-to-use interface (Streamlit UI)
2. Real-time prediction
3. Supports multiple input parameters
4. Accurate and fast results
5. Helps in industrial decision-making
             
---

### 🔹 Future Scope
1. Add color prediction model
2. Deploy as a web application
3. Integrate real-time sensor data
4. Improve model accuracy with real datasets
5. Add recommendation system for optimal formulation

    ---
    Go to the **Prediction** page to try out the viscosity prediction system!
    """)

# ===============================
# 📈 PREDICTION PAGE
# ===============================
elif page == "📈 Prediction":
    st.title("🧪 Paint Viscosity Prediction")

    st.write("Enter values and click **Predict Viscosity**.")

    col1, col2 = st.columns(2)

    with col1:
        pigment_type = st.selectbox("Pigment Type", ["titanium", "ironoxide", "carbonblack"])
        binder_type = st.selectbox("Binder Type", ["acrylic", "alkyd", "epoxy"])
        solvent_type = st.selectbox("Solvent Type", ["water", "xylene", "toluene"])
        additive_type = st.selectbox("Additive Type", ["dispersant", "thickener", "stabilizer"])

        pigment_percent = st.number_input("Pigment Percent(range: 10.0 - 30.0)", 10.0, 30.0, 15.0)
        binder_percent = st.number_input("Binder Percent(range: 20.0 - 41.0)", 20.0, 41.0, 30.0)
        solvent_percent = st.number_input("Solvent Percent(range: 25.5 - 48.0)", 25.5, 48.0, 35.0)
        additive_percent = st.number_input("Additive Percent(range: 0.8 - 5.5)", 0.8, 5.5, 2.0)
        solid_content_percent = st.number_input("Solid Content Percent(range: 40.0 - 67.8)", 40.0, 67.8, 50.0)

        particle_size_um = st.number_input("Particle Size(range: 2.0 - 13.3)", 2.0, 13.3, 5.0)
        resin_viscosity = st.number_input("Resin Viscosity(range: 420.0 - 1500.0)", 420.0, 1500.0, 800.0)
        density_g_cm3 = st.number_input("Density(range: 1.21 - 1.50)", 1.21, 1.50, 1.30)
        ph = st.number_input("pH(range: 5.8 - 9.0)", 5.8, 9.0, 7.0)

    with col2:
        temperature_c = st.number_input("Temperature(range: 18.0 - 42.0)", 18.0, 42.0, 25.0)
        pressure_kpa = st.number_input("Pressure(range: 90.0 - 125.0)", 90.0, 125.0, 100.0)
        humidity_percent = st.number_input("Humidity(range: 25.0 - 80.0)", 25.0, 80.0, 50.0)

        mixing_speed_rpm = st.number_input("Mixing Speed(range: 850.0 - 1600.0)", 850.0, 1600.0, 1000.0)
        mixing_time_min = st.number_input("Mixing Time(range: 20.0 - 60.0)", 20.0, 60.0, 30.0)
        batch_size_l = st.number_input("Batch Size(range: 200.0 - 1000.0)", 200.0, 1000.0, 500.0)
        cooling_time_min = st.number_input("Cooling Time(range: 10.0 - 40.0)", 10.0, 40.0, 20.0)
        drying_time_min = st.number_input("Drying Time(range: 35.0 - 105.0)", 35.0, 105.0, 60.0)

        gloss_level = st.selectbox("Gloss Level", ["low", "medium", "high"])
        color = st.selectbox(
            "Color",
            [
                "Amber Orange","Apricot Orange","Ash Gray","Brick Red","Burgundy Red",
                "Burnt Orange","Canary Yellow","Caramel Brown","Charcoal Black","Chestnut Brown",
                "Chocolate Brown","Cloud White","Cobalt Blue","Coral Orange","Crimson Red",
                "Ebony Black","Emerald Green","Forest Green","Golden Yellow","Graphite Black",
                "Honey Yellow","Ivory White","Jet Black","Lemon Yellow","Lime Green",
                "Metallic Silver","Moon Silver","Mustard Yellow","Navy Blue","Olive Green",
                "Onyx Black","Pearl Gray","Pearl White","Platinum Silver","Pure White",
                "Rose Red","Sage Green","Sapphire Blue","Scarlet Red","Sky Blue",
                "Slate Gray","Smoke Gray","Snow White","Steel Gray","Steel Silver",
                "Sterling Silver","Tangerine Orange","Teal Blue","Umber Brown","Walnut Brown"
            ]
        ).lower()

    # -------------------------------
    # Prediction
    # -------------------------------
    if st.button("Predict Viscosity"):
        try:
            sample = {
                'pigment_type': pigment_type,
                'binder_type': binder_type,
                'solvent_type': solvent_type,
                'additive_type': additive_type,
                'pigment_percent': pigment_percent,
                'binder_percent': binder_percent,
                'solvent_percent': solvent_percent,
                'additive_percent': additive_percent,
                'solid_content_percent': solid_content_percent,
                'particle_size_um': particle_size_um,
                'resin_viscosity': resin_viscosity,
                'density_g_cm3': density_g_cm3,
                'ph': ph,
                'temperature_c': temperature_c,
                'pressure_kpa': pressure_kpa,
                'humidity_percent': humidity_percent,
                'mixing_speed_rpm': mixing_speed_rpm,
                'mixing_time_min': mixing_time_min,
                'batch_size_l': batch_size_l,
                'cooling_time_min': cooling_time_min,
                'drying_time_min': drying_time_min,
                'gloss_level': gloss_level,
                'color': color
            }

            sample_df = pd.DataFrame([sample])
            sample_df = pd.get_dummies(sample_df)
            sample_df = sample_df.reindex(columns=columns, fill_value=0)

            sample_scaled = scaler.transform(sample_df)
            pred_value = float(model.predict(sample_scaled)[0])

            # -------------------------------
            # Viscosity level + usage + recommendation + suggestions
            # -------------------------------
            if pred_value < 900:
                viscosity_level = "🟢 Low Viscosity"
                usage_type = "Primer"
                recommended_usage = (
                    "This paint is mainly recommended for primer and base-coat applications where a thinner and "
                    "easier-flowing formulation is needed. It can be applied on wood, metal, and plastic surfaces "
                    "before the final top coat to improve surface bonding, reduce uneven absorption, and create a "
                    "smooth foundation for the next layer of paint. This type of viscosity is useful when the goal "
                    "is to achieve better adhesion, uniform spreading, and proper surface preparation without making "
                    "the coating too thick."
                )
                suggestions = (
                    "Since the paint is too thin, increase the binder and solid content to improve thickness and strength, reduce the solvent percentage to avoid excessive dilution, slightly decrease the mixing speed to prevent over-thinning, and add suitable thickening additives to achieve better consistency and coverage."
                )

            elif pred_value < 1200:
                viscosity_level = "🟡 Medium Viscosity"
                usage_type = "Interior"
                recommended_usage = (
                    "This paint is mainly recommended for interior coating applications such as walls, ceilings, "
                    "furniture, wooden panels, and selected plastic surfaces where a balanced flow and finish are "
                    "required. This viscosity range generally supports good spreadability, smooth application, and an "
                    "attractive decorative appearance, making it suitable for indoor environments. It is especially "
                    "useful in places where visual quality, neat finish, and comfortable aesthetics are important, "
                    "such as homes, offices, and indoor commercial spaces."
                )
                suggestions = (
                    "Since the paint has balanced viscosity, maintain the current composition and process parameters, ensure proper mixing time for uniformity, control temperature and humidity during production, and perform regular quality checks to retain consistent and optimal performance."
                )

            elif pred_value < 1500:
                viscosity_level = "🟠 High Viscosity"
                usage_type = "Exterior"
                recommended_usage = (
                    "This paint is mainly recommended for exterior coating applications where a thicker and more "
                    "protective film is needed. It can be used on outside building walls, metal structures, wooden "
                    "surfaces, boundary areas, and other exposed sections that face environmental stress. This "
                    "viscosity range is helpful for creating coatings that provide better coverage, stronger surface "
                    "protection, and improved resistance against weather conditions such as sunlight, rain, moisture, "
                    "and temperature variation. It is suitable where durability is more important than a very light flow."
                )
                suggestions = (
                    "Since the paint is thick and difficult to apply, increase the solvent percentage to improve flow, reduce solid content to lower thickness, increase mixing speed and time for better uniformity, and slightly adjust temperature conditions to make the paint easier to spread."
                )

            else:
                viscosity_level = "🔴 Very High Viscosity"
                usage_type = "Industrial"
                recommended_usage = (
                    "This paint is mainly recommended for industrial and heavy-duty coating applications where a very "
                    "strong, thick, and durable layer is required. It can be used on machinery, pipelines, industrial "
                    "metal equipment, factory surfaces, structural components, and other demanding environments where "
                    "high resistance is important. This viscosity range is suitable for applications that require better "
                    "mechanical strength, stronger coating build, and improved resistance to wear, chemicals, and harsh "
                    "working conditions over time."
                )
                suggestions = (
                    "Since the paint is extremely thick, significantly increase the solvent content to reduce heaviness, decrease binder and solid percentages to balance the formulation, increase mixing speed and duration for proper dispersion, optimize temperature conditions, and use flow-enhancing additives to improve usability and application quality."
                )

            # -------------------------------
            # 🎯 Prediction Result
            # -------------------------------
            st.markdown("## 🎯 Prediction Result")

            # 1st row - Predicted Viscosity
            st.metric(label="🧪 Predicted Viscosity", value=f"{pred_value:.2f} cP")

            # 2nd row - Viscosity Level
            if pred_value < 900:
                st.success(viscosity_level)
            elif pred_value < 1200:
                st.info(viscosity_level)
            elif pred_value < 1500:
                st.warning(viscosity_level)
            else:
                st.error(viscosity_level)

            # 3rd row - Usage Type
            st.metric(label="🏷️ Usage Type", value=usage_type)

            # 4th row - Recommended Usage
            st.subheader("✅ Recommended Usage")
            st.info(recommended_usage)

            # 5th row - Suggestions
            st.subheader("💡 Suggestions")
            st.warning(suggestions)

        except Exception as e:
            st.error(f"Error: {e}")