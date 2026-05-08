import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="USD Exchange Rate Prediction",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""

<style>

/* HEADER */

.stApp {
    background: #111518;
    color: white;
    font-family: Arial, sans-serif;
}

.block-container {
    padding-top: 0.5rem !important;
    padding-left: 1.2rem !important;
    padding-right: 1.2rem !important;
    max-width: 100% !important;
}

/* КНОПКА ОТКРЫТИЯ SIDEBAR */

button[kind="header"] {
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 999999 !important;
    background: #171c20 !important;
    border: 1px solid #2b333a !important;
    border-radius: 10px !important;
    width: 42px !important;
    height: 42px !important;
    font-size: 0px !important;
    color: transparent !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.35);
}

/* ПОЛНОСТЬЮ СКРЫВАЕМ ВСЕ ВНУТРИ */

button[kind="header"] * {
    display: none !important;
}

/* СВОЯ ИКОНКА */

button[kind="header"]::after {
    content: "☰";
    color: white;
    font-size: 24px;
    font-weight: 900;
    position: absolute;
    top: 7px;
    left: 11px;
}

button[kind="header"]:hover {
    background: #222931 !important;
    border: 1px solid #ff9f1c !important;
}

/* SIDEBAR */

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14181d 0%, #0d1014 100%);
    width: 185px !important;
    border-right: 1px solid #242b31;
    transition: all 0.3s ease;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-logo {
    text-align: center;
    padding-top: 10px;
    padding-bottom: 20px;
}

.logo-icon {
    font-size: 38px;
}

.logo-title {
    font-size: 15px;
    font-weight: 900;
    margin-top: 6px;
}

.logo-sub {
    font-size: 10px;
    color: #9aa3ad !important;
}

.main-title {
    font-size: 26px;
    font-weight: 900;
    color: white;
    margin-bottom: 0px;
}

.subtitle {
    font-size: 12px;
    color: #9aa3ad;
    margin-bottom: 14px;
}

.title-box {
    background: #171c20;
    border: 1px solid #2b333a;
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 15px;
    font-size: 14px;
    font-weight: 900;
    color: white;
}

.chart-card {
    background: #171c20;
    border-radius: 12px;
    padding: 10px;
    margin-top: 10px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(90deg, #ff9f1c, #ffb238);
    border: none;
    color: white;
    border-radius: 8px;
    height: 40px;
    font-weight: 900;
    font-size: 13px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #ffb238, #ff9f1c);
    color: white;
}

div[data-baseweb="select"] > div {
    background-color: #101417 !important;
    border: 1px solid #303941 !important;
    color: white !important;
}

input {
    background-color: #101417 !important;
    color: white !important;
}

.result-label {
    text-align: center;
    color: #b9c1c9;
    font-size: 13px;
}

.result-country {
    text-align: center;
    font-size: 24px;
    font-weight: 900;
    margin-top: 10px;
}

.result-number {
    text-align: center;
    font-size: 40px;
    font-weight: 900;
    color: #ff9f1c;
    margin-top: 10px;
}

.result-sub {
    text-align: center;
    color: #b9c1c9;
    font-size: 13px;
}

.green-badge {
    width: fit-content;
    margin: 14px auto 0 auto;
    background: rgba(34,197,94,0.15);
    color: #34d37a;
    padding: 8px 14px;
    border-radius: 8px;
    font-size: 11px;
    font-weight: 900;
}

.footer-note {
    margin-top: 8px;
    color: #9aa3ad;
    font-size: 11px;
}

</style>

""", unsafe_allow_html=True)

st.sidebar.markdown(
"""
<div style="text-align:center; margin-top:15px;"></div>
""",
unsafe_allow_html=True
)

st.sidebar.image(
    "images/logo.png",
    width=180
)

st.sidebar.markdown(
"""
<div style="
     text-align:center;
     font-size:28px;
     font-weight:900;
     color:white;
     margin-top:8px;
">
USD EXCHANGE
</div>

<div style="
    text-align:center;
    color:#8ea0b5;
    font-size:13px;
    letter-spacing:2px;
    margin-bottom:25px;
">
    PREDICTOR
</div>
""",
unsafe_allow_html=True
)

page = st.sidebar.radio(
    "",
    ["🏠 Prediction", "🌐 Countries", "👤 About", "ℹ️ Data Info"]
)

countries = [
    "Australia", "Canada", "Denmark", "Egypt", "Germany",
    "India", "Indonesia", "Japan", "Kazakhstan",
    "Kyrgyzstan", "Mexico", "Russia",
    "Thailand", "Turkiye", "Vietnam"
]

currency_map = {
    "Australia": "AUD",
    "Canada": "CAD",
    "Denmark": "DKK",
    "Egypt": "EGP",
    "Germany": "EUR",
    "India": "INR",
    "Indonesia": "IDR",
    "Japan": "JPY",
    "Kazakhstan": "KZT",
    "Kyrgyzstan": "KGS",
    "Mexico": "MXN",
    "Russia": "RUB",
    "Thailand": "THB",
    "Turkiye": "TRY",
    "Vietnam": "VND"
}

feature_names = {
    "ER_lag1": "Previous Year Exchange Rate",
    "INF": "Inflation Rate (%)",
    "INF_lag1": "Previous Year Inflation Rate",
    "GDP": "GDP Growth (%)",
    "GDP_lag1": "Previous Year GDP Growth",
    "TRADEB": "Trade Balance",
    "TRADEB_lag1": "Previous Year Trade Balance",
    "ACCB_lag1": "Previous Year Current Account Balance",
    "INDPROD_lag1": "Previous Year Industrial Production",
    "FFR_lag1": "Previous Year Federal Funds Rate"
}

if page == "🏠 Prediction":

    st.markdown(
        '<div class="main-title">USD Exchange Rate Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Multiple Linear Regression Models for 15 Countries</div>',
        unsafe_allow_html=True
    )

    left, right = st.columns([1, 1], gap="medium")

    with left:

        st.markdown(
            '<div class="title-box">1. Choose Country</div>',
            unsafe_allow_html=True
        )

        country = st.selectbox(
            "Select a country",
            [
                "🇦🇺 Australia",
                "🇨🇦 Canada",
                "🇩🇰 Denmark",
                "🇪🇬 Egypt",
                "🇩🇪 Germany",
                "🇮🇳 India",
                "🇮🇩 Indonesia",
                "🇯🇵 Japan",
                "🇰🇿 Kazakhstan",
                "🇰🇬 Kyrgyzstan",
                "🇲🇽 Mexico",
                "🇷🇺 Russia",
                "🇹🇭 Thailand",
                "🇹🇷 Turkiye",
                "🇻🇳 Vietnam"
            ]
        )

        country = country.split(" ", 1)[1]

        flag_code_map = {
            "Australia": "au",
            "Canada": "ca",
            "Denmark": "dk",
            "Egypt": "eg",
            "Germany": "de",
            "India": "in",
            "Indonesia": "id",
            "Japan": "jp",
            "Kazakhstan": "kz",
            "Kyrgyzstan": "kg",
            "Mexico": "mx",
            "Russia": "ru",
            "Thailand": "th",
            "Turkiye": "tr",
            "Vietnam": "vn"
        }

        flag_url = f"https://flagcdn.com/w160/{flag_code_map[country]}.png"

        st.markdown(
            f"""
            <div style="
                display:flex;
                align-items:center;
                gap:14px;
                background:#171c20;
                border:1px solid #2b333a;
                border-radius:12px;
                padding:12px 16px;
                margin-top:10px;
                margin-bottom:18px;
            ">
                <img src="{flag_url}" style="
                    width:58px;
                    height:38px;
                    object-fit:cover;
                    border-radius:6px;
                    box-shadow:0 4px 12px rgba(0,0,0,0.35);
                ">
                <div style="
                    font-size:18px;
                    font-weight:900;
                    color:white;
                ">
                    {country}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        model_path = f"models/{country}_model.pkl"

        if os.path.exists(model_path):
            model_package = joblib.load(model_path)
            features = model_package["features"]
            model_loaded = True
        else:
            model_loaded = False
            features = []

        st.markdown(
            '<div class="title-box" style="margin-top:18px;">2. Enter Economic Indicators</div>',
            unsafe_allow_html=True
        )

        user_inputs = {}

        if model_loaded:

            for feature in features:

                label = feature_names.get(feature, feature)

                value = st.text_input(label, value="0")

                try:
                    user_inputs[feature] = float(value)
                except:
                    user_inputs[feature] = 0.0

        predict_button = st.button("↗ Predict Exchange Rate")

    with right:

        st.markdown(
            '<div class="title-box">3. Prediction Result</div>',
            unsafe_allow_html=True
        )

        prediction = None

        if predict_button and model_loaded:

            model = model_package["model"]
            scaler = model_package["scaler"]

            input_values = [[user_inputs[f] for f in features]]

            input_scaled = scaler.transform(input_values)

            prediction = model.predict(input_scaled)[0]

        currency = currency_map[country]

        if prediction is not None:

            st.markdown(
                f"""
                <div class="result-label">Predicted USD Exchange Rate for</div>
                <div class="result-country">{country}</div>
                <div class="result-number">{prediction:.2f} {currency}</div>
                <div class="result-sub">({currency} per 1 USD)</div>
                <div class="green-badge">
                    ↑ Model prediction based on selected factors
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                f"""
                <div class="result-label">Predicted USD Exchange Rate for</div>
                <div class="result-country">{country}</div>
                <div class="result-number">---</div>
                <div class="result-sub">
                    Enter indicators and click prediction button
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown(
            '<div class="title-box" style="margin-top:18px;">4. Historical Exchange Rate Trend (2000–2024)</div>',
            unsafe_allow_html=True
        )

        data_path = "DIPLOM_DATA_16.xlsx"

        if os.path.exists(data_path):

            df = pd.read_excel(data_path)

            df_country_chart = df[
                (df["COUNTRY"] == country) &
                (df["YEAR"] >= 2000) &
                (df["YEAR"] <= 2024)
            ].copy()

            df_country_chart = df_country_chart.sort_values("YEAR")

            if not df_country_chart.empty:

                fig = px.line(
                    df_country_chart,
                    x="YEAR",
                    y="ER",
                    markers=True
                )

                fig.update_traces(
                    line_color="#ff9f1c",
                    line_width=3,
                    marker=dict(size=8),
                    hovertemplate=
                    "<b>Year:</b> %{x}<br>" +
                    "<b>Exchange Rate:</b> %{y:.2f}<extra></extra>"
                )

                fig.update_layout(
                    paper_bgcolor="#171c20",
                    plot_bgcolor="#171c20",
                    font_color="white",
                    margin=dict(l=10, r=10, t=10, b=10),

                    xaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.08)"
                    ),

                    yaxis=dict(
                        showgrid=True,
                        gridcolor="rgba(255,255,255,0.08)"
                    )
                )

                st.plotly_chart(fig, use_container_width=True)

            else:

                st.warning(f"No data found for {country}")

        else:

            st.warning("DIPLOM_DATA_16.xlsx not found")

    st.markdown(
        '<div class="footer-note">ⓘ Historical exchange rate data from 2000 to 2024</div>',
        unsafe_allow_html=True
    )

elif page == "🌐 Countries":

    st.markdown('<div class="main-title">Countries</div>', unsafe_allow_html=True)

    st.markdown("""
<div style="color:#b9c1c9; font-size:15px; line-height:1.7; margin-top:15px; margin-bottom:25px;">
    This section presents the 15 countries included in the exchange rate prediction model.
    Each country is described by its region, national currency, and general economic profile.
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<style>
.country-card {
    background:#171c20;
    border:1px solid #2b333a;
    border-radius:22px;
    padding:30px;
    margin-bottom:22px;
    min-height:330px;
    transition:0.35s ease;
    box-shadow:0 8px 24px rgba(0,0,0,0.25);
}

.country-card:hover {
    transform:translateY(-6px);
    border-color:#ff9f1c;
    box-shadow:0 18px 40px rgba(255,159,28,0.18);
}

.country-flag-img {
    width:110px;
    height:75px;
    object-fit:cover;
    border-radius:10px;
    margin-bottom:20px;
    box-shadow:0 6px 18px rgba(0,0,0,0.25);
}

.country-name {
    font-size:30px;
    font-weight:900;
    color:white;
    margin-bottom:18px;
}

.country-meta {
    color:#d8dee6;
    font-size:15px;
    line-height:1.9;
    margin-bottom:16px;
}

.country-desc {
    color:#b9c1c9;
    font-size:14px;
    line-height:1.8;
    margin-top:18px;
}

.currency-badge {
    display:inline-block;
    background:rgba(255,159,28,0.14);
    color:#ffb238;
    padding:6px 14px;
    border-radius:18px;
    font-size:12px;
    font-weight:800;
    margin-top:10px;
}
</style>
""", unsafe_allow_html=True)



    country_cards = [

        {
        "flag": "au",
        "name": "Australia",
        "region": "Oceania",
        "currency": "Australian Dollar (AUD)",
        "description": "Australia is a highly developed and stable economy located in the Oceania region. The country is one of the world's leading exporters of iron ore, coal, gold, and natural gas. Its economy is strongly connected to global commodity markets and international trade, especially with Asian countries such as China and Japan. Australia also has a strong banking system, high living standards, and advanced financial infrastructure, making the Australian dollar an important currency in global exchange markets."
        },

        {
        "flag": "ca",
        "name": "Canada",
        "region": "North America",
        "currency": "Canadian Dollar (CAD)",
        "description": "Canada is one of the largest developed economies in North America with strong oil, gas, mining, and manufacturing industries. The Canadian economy is closely connected to the United States through trade and financial markets. Canada is known for political stability, high-quality infrastructure, and advanced banking systems. The Canadian dollar is often influenced by energy prices and global commodity demand because the country is a major exporter of natural resources."
        },

        {
        "flag": "dk",
        "name": "Denmark",
        "region": "Northern Europe",
        "currency": "Danish Krone (DKK)",
        "description": "Denmark is a high-income European country with a strong social welfare system and stable economic environment. The economy is supported by international trade, renewable energy production, pharmaceuticals, shipping, and technology industries. Denmark is considered one of the most financially stable countries in Europe and has low levels of corruption and strong institutional quality."
        },

        {
        "flag": "eg",
        "name": "Egypt",
        "region": "North Africa",
        "currency": "Egyptian Pound (EGP)",
        "description": "Egypt is one of the largest economies in North Africa and plays an important strategic role due to the Suez Canal, which connects global trade routes. The economy is supported by tourism, agriculture, construction, energy, and international trade. Egypt has experienced significant economic reforms and infrastructure development in recent years, making it an important emerging market economy in the region."
        },

        {
        "flag": "de",
        "name": "Germany",
        "region": "Western Europe",
        "currency": "Euro (EUR)",
        "description": "Germany is one of the world's largest industrial and export-oriented economies. It is known for advanced engineering, automotive manufacturing, machinery production, and strong international trade relations. Germany plays a leading role in the European Union and has a highly developed financial system, modern infrastructure, and strong technological innovation. The country's economy significantly influences the European and global financial markets."
        },

        {
        "flag": "in",
        "name": "India",
        "region": "South Asia",
        "currency": "Indian Rupee (INR)",
        "description": "India is one of the fastest-growing major economies in the world with strong development in information technology, manufacturing, services, and telecommunications. The country has a very large population and labor force, which supports domestic economic growth and consumer demand. India is becoming increasingly important in global trade and international investment markets."
        },

        {
        "flag": "id",
        "name": "Indonesia",
        "region": "Southeast Asia",
        "currency": "Indonesian Rupiah (IDR)",
        "description": "Indonesia is the largest economy in Southeast Asia and is rich in natural resources such as coal, palm oil, gas, and minerals. The country has a growing manufacturing sector, expanding middle class, and increasing foreign investment activity. Indonesia is an important emerging market economy with strong regional trade connections."
        },

        {
        "flag": "jp",
        "name": "Japan",
        "region": "East Asia",
        "currency": "Japanese Yen (JPY)",
        "description": "Japan is a highly industrialized and technologically advanced economy known for automotive production, robotics, electronics, and financial services. The country has one of the largest GDPs in the world and plays an important role in global trade and finance. The Japanese yen is considered one of the major reserve currencies and is often viewed as a safe-haven currency during periods of global uncertainty."
        },

        {
        "flag": "kz",
        "name": "Kazakhstan",
        "region": "Central Asia",
        "currency": "Kazakhstani Tenge (KZT)",
        "description": "Kazakhstan is the largest economy in Central Asia and is rich in oil, natural gas, uranium, and mineral resources. The economy is strongly influenced by global commodity prices and export revenues. Kazakhstan plays an important regional role in energy transportation and international trade between Asia and Europe."
        },

        {
        "flag": "kg",
        "name": "Kyrgyzstan",
        "region": "Central Asia",
        "currency": "Kyrgyzstani Som (KGS)",
        "description": "Kyrgyzstan is a developing economy in Central Asia where agriculture, mining, hydropower, and remittances from foreign workers are important economic sectors. The country has growing trade relations with neighboring countries and regional economic organizations."
        },

        {
        "flag": "mx",
        "name": "Mexico",
        "region": "North America",
        "currency": "Mexican Peso (MXN)",
        "description": "Mexico is one of the largest emerging economies in Latin America with strong manufacturing, automotive, oil, and export industries. The country has close economic integration with the United States and Canada through regional trade agreements. Mexico's economy is strongly connected to international trade and industrial production."
        },

        {
        "flag": "ru",
        "name": "Russia",
        "region": "Eastern Europe / Northern Asia",
        "currency": "Russian Ruble (RUB)",
        "description": "Russia is one of the world's largest energy-exporting economies with major oil, gas, and natural resource industries. The Russian economy is highly affected by global commodity prices, international trade, and geopolitical developments. Russia also has large industrial, transportation, and agricultural sectors."
        },

        {
        "flag": "th",
        "name": "Thailand",
        "region": "Southeast Asia",
        "currency": "Thai Baht (THB)",
        "description": "Thailand has a diversified economy supported by tourism, manufacturing, agriculture, and exports. The country is an important production center for automobiles, electronics, and food products in Southeast Asia. Tourism plays a major role in Thailand's economic growth and foreign currency earnings."
        },

        {
        "flag": "tr",
        "name": "Turkiye",
        "region": "Western Asia / Southeast Europe",
        "currency": "Turkish Lira (TRY)",
        "description": "Turkiye has a strategically important economy connecting Europe and Asia through trade, transportation, and logistics. The economy includes strong manufacturing, construction, tourism, and export sectors. The Turkish economy is influenced by inflation, monetary policy, and international investment flows."
        },

        {
        "flag": "vn",
        "name": "Vietnam",
        "region": "Southeast Asia",
        "currency": "Vietnamese Dong (VND)",
        "description": "Vietnam is one of the fastest-growing economies in Southeast Asia with rapid industrialization and export expansion. The country has become an important global manufacturing center for electronics, textiles, and technology products. Vietnam continues to attract large amounts of foreign direct investment due to its competitive labor market and economic growth."
        }

    ]

    for i in range(0, len(country_cards), 2):
        col1, col2 = st.columns(2)

        for col, country in zip([col1, col2], country_cards[i:i+2]):
            with col:

                flag_url = f"https://flagcdn.com/w160/{country['flag']}.png"

                card_html = (
                    '<div class="country-card">'
                    f'<img src="{flag_url}" class="country-flag-img">'
                    f'<div class="country-name">{country["name"]}</div>'
                    '<div class="country-meta">'
                    f'<span style="color:white; font-weight:800;">Region:</span> {country["region"]}<br><br>'
                    f'<span style="color:white; font-weight:800;">Currency:</span> {country["currency"]}'
                    '</div>'
                    '<div class="currency-badge">Exchange Rate Model</div>'
                    f'<div class="country-desc">{country["description"]}</div>'
                    '</div>'
                )

                st.markdown(card_html, unsafe_allow_html=True)


elif page == "👤 About":

    import base64
    from pathlib import Path

    def image_to_base64(path):
        img_path = Path(path)
        if not img_path.exists():
            return ""
        return base64.b64encode(img_path.read_bytes()).decode()

    ayazhan_img = image_to_base64("images/Ayazhan.jpg")
    duman_img = image_to_base64("images/Duman.jpg")
    aykerim_img = image_to_base64("images/Aykerim.jpg")

    st.markdown('<div class="main-title">About Project</div>', unsafe_allow_html=True)

    st.markdown('<div class="title-box" style="margin-top:25px;">🌐 About Website</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:20px; color:#d8dee6; font-size:15px; line-height:1.8; margin-bottom:20px;">
        This web application was developed to predict the U.S. dollar exchange rate
        using machine learning and macroeconomic indicators.
        <br><br>
        Users can select a country, enter economic indicators,
        and receive a predicted exchange rate based on the trained model.
        <br><br>
        The platform also provides historical exchange rate trends
        from 2000 to 2024 for each selected country.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-box">⚙️ Main Features</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:18px; margin-bottom:15px;">
            <div style="font-size:17px; font-weight:900; color:white;">📈 Exchange Rate Prediction</div>
            <div style="margin-top:10px; color:#b9c1c9; line-height:1.7;">
                Predicts exchange rates using economic indicators and machine learning models.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:18px;">
            <div style="font-size:17px; font-weight:900; color:white;">📊 Multiple Linear Regression</div>
            <div style="margin-top:10px; color:#b9c1c9; line-height:1.7;">
                The application uses Multiple Linear Regression models
                to analyze the relationship between macroeconomic indicators
                and USD exchange rates for 15 countries.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:18px; margin-bottom:15px;">
            <div style="font-size:17px; font-weight:900; color:white;">📊 Historical Visualization</div>
            <div style="margin-top:10px; color:#b9c1c9; line-height:1.7;">
                Interactive charts display historical exchange rate trends from 2000–2024.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:18px;">
            <div style="font-size:17px; font-weight:900; color:white;">🤖 Machine Learning</div>
            <div style="margin-top:10px; color:#b9c1c9; line-height:1.7;">
                Built using machine learning techniques,
                data preprocessing, visualization, and macroeconomic data analysis.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="title-box" style="margin-top:25px;">👨‍💻 Meet the Team</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:20px; color:#d8dee6; font-size:15px; line-height:1.9; margin-bottom:25px;">
        This project was developed as part of a diploma research project
        focused on exchange rate prediction and economic analysis.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .creator-card {
        background:#171c20;
        border:1px solid #2b333a;
        border-radius:22px;
        padding:22px;
        text-align:center;
        transition:0.35s ease;
        box-shadow:0 8px 24px rgba(0,0,0,0.25);
        min-height:620px;
    }

    .creator-card:hover {
        transform:translateY(-6px);
        border-color:#ff9f1c;
        box-shadow:0 18px 40px rgba(255,159,28,0.18);
    }

    .creator-photo {
        width:100%;
        height:430px;
        object-fit:cover;
        border-radius:18px;
        transition:0.35s ease;
    }

    .creator-card:hover .creator-photo {
        transform:scale(1.02);
    }

    .creator-name {
        font-size:24px;
        font-weight:900;
        color:white;
        text-align:center;
        margin-top:20px;
    }

    .creator-role {
        color:#b9c1c9;
        text-align:center;
        font-size:15px;
        margin-top:8px;
        margin-bottom:18px;
    }

    .creator-badge {
        background:rgba(255,159,28,0.14);
        color:#ffb238;
        padding:7px 14px;
        border-radius:20px;
        font-size:13px;
        font-weight:800;
        text-align:center;
        width:fit-content;
        margin:0 auto 20px auto;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="creator-card">
            <img src="data:image/jpg;base64,{ayazhan_img}" class="creator-photo">
            <div class="creator-name">Otegen Ayazhan</div>
            <div class="creator-role">Data Analysis & Literature</div>
            <div class="creator-badge">Economic Research</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="creator-card">
            <img src="data:image/jpg;base64,{duman_img}" class="creator-photo">
            <div class="creator-name">Assylkhanov Duman</div>
            <div class="creator-role">Data Analysis & ML</div>
            <div class="creator-badge">Data Analytics</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="creator-card">
            <img src="data:image/jpg;base64,{aykerim_img}" class="creator-photo">
            <div class="creator-name">Zhumagulova Aykerim</div>
            <div class="creator-role">Data Analysis & Research</div>
            <div class="creator-badge">Data Analytics</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#171c20; border:1px solid #2b333a; border-radius:12px; padding:18px; color:#d8dee6; font-size:15px; line-height:1.8; margin-top:25px; text-align:center;">
        <b>University:</b> SDU University
    </div>
    """, unsafe_allow_html=True)

elif page == "ℹ️ Data Info":

    st.markdown('<div class="main-title">Data Information</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="font-size:16px; color:#d8dee6; margin-top:18px; line-height:1.7;">
        The dataset used in this project was collected from official and reliable sources,
        mainly from the <b>World Bank Open Data</b>. These indicators were selected because
        they can influence the exchange rate of the U.S. dollar in different countries.
        <br><br>
        Source: 
        <a href="https://data.worldbank.org/" target="_blank" style="color:#ff9f1c; text-decoration:none;">
            World Bank Open Data
        </a>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="title-box" style="margin-top:25px;">Description of Variables</div>', unsafe_allow_html=True)

    data_info = {
    "Exchange Rate": "The price of the U.S. dollar expressed in the domestic currency of each country.",
    
    "Consumer Price Index (CPI)": "Shows the annual percentage change in the general price level of goods and services. It is used to measure inflation.",
    
    "Unemployment Rate": "The percentage of the labor force that is unemployed and actively seeking work.",
    
    "GDP Growth": "Annual percentage growth rate of gross domestic product at constant prices. It shows the economic growth of a country.",
    
    "Control of Corruption": "An indicator measuring the extent to which public power is exercised for private gain.",
    
    "Foreign Reserves": "External assets held by the central bank, including foreign currencies and securities.",
    
    "Trade Balance": "The difference between exports and imports of goods and services.",
    
    "Current Account Balance": "Net flow of goods, services, income, and transfers between countries.",
    
    "Industrial Production": "Measures output in industrial sectors such as manufacturing, mining, and utilities.",
    
    "Oil Price per Barrel": "The global market price of crude oil. It can affect exchange rates, especially in countries dependent on oil exports or imports.",
    
    "Federal Funds Rate": "The interest rate at which U.S. banks lend reserves to each other overnight. It can influence global capital flows and the U.S. dollar exchange rate.",
    
    "Gold Price": "The international market price of gold. It is often used as a safe-haven asset and may affect currency movements."
}

    for variable, description in data_info.items():
        st.markdown(f"""
        <div style="
            background:#171c20;
            border:1px solid #2b333a;
            border-radius:10px;
            padding:14px 16px;
            margin-bottom:10px;
        ">
            <div style="font-size:16px; font-weight:900; color:white;">
                {variable}
            </div>
            <div style="font-size:14px; color:#b9c1c9; margin-top:6px; line-height:1.5;">
                {description}
            </div>
        </div>
        """, unsafe_allow_html=True)