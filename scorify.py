import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_models():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    model_conv = joblib.load(os.path.join(BASE_DIR, 'model_conv.pkl'))
    model_profit = joblib.load(os.path.join(BASE_DIR, 'model_profit.pkl'))
    encoders = joblib.load(os.path.join(BASE_DIR, 'encoders.pkl'))
    features = joblib.load(os.path.join(BASE_DIR, 'feature_names.pkl'))
    return model_conv, model_profit, encoders, features

model_conv, model_profit, encoders, feature_names = load_models()

def raw_to_score(raw_prob):
    raw_prob = max(0, min(raw_prob, 1.0))
    if raw_prob <= 0.01:
        score = (raw_prob / 0.01) * 20
    elif raw_prob <= 0.05:
        score = 20 + ((raw_prob - 0.01) / (0.05 - 0.01)) * 30
    elif raw_prob <= 0.25:
        score = 50 + ((raw_prob - 0.05) / (0.25 - 0.05)) * 30
    else:
        score = 80 + ((raw_prob - 0.25) / (0.75 - 0.25)) * 20
    return min(100, max(0, score))

st.set_page_config(page_title="Scorify", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .main-title { font-size: 2.5rem; font-weight: bold; margin-bottom: 0; }
    .sub-title { font-size: 1.1rem; color: #666; margin-top: 0; }
    .score-box { padding: 20px; border-radius: 10px; text-align: center; }
    .top { background-color: #ffebee; border: 2px solid #c62828; }
    .high { background-color: #fff3e0; border: 2px solid #e65100; }
    .medium { background-color: #e8f5e9; border: 2px solid #2e7d32; }
    .low { background-color: #e3f2fd; border: 2px solid #1565c0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">📊 SCORIFY</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Lead Scoring — Enter lead details to get real-time predictions</p>', unsafe_allow_html=True)
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Lead Profile")
    travel_group = st.selectbox("Travel Group Type", ['PARTNER', 'FAMILY', 'FRIENDS', 'SOLO', 'Unknown'])
    trip_duration = st.selectbox("Trip Duration", ['10 Days', 'One Week', '2 Weeks', '2 Weeks +', 'Less than One Week', 'Unknown'])
    budget_level = st.selectbox("Budget Level", ['low', 'med', 'high', 'notsure', 'Unknown'])
    budget_range = st.selectbox("Budget Range", ['4000 - 7000 USD', '7000 - 10000 USD', '5000 - 7500 USD', '2500 - 5000 USD', '10000+ USD', 'NOTSURE', 'Unknown'])
    currency = st.selectbox("Currency", ['USD', 'EUR', 'GBP', 'AUD', 'CAD', 'ZAR', 'Unknown'])
    age_bracket = st.selectbox("Age Bracket", ['35-54', '55+', '18-34', 'Unknown'])
    country = st.selectbox("Country", ['United States', 'Australia', 'Canada', 'United Kingdom', 'Germany',
                                        'New Zealand', 'Netherlands', 'India', 'Italy', 'France', 'South Africa',
                                        'Singapore', 'Switzerland', 'Ireland', 'Sweden', 'Unknown'])

with col2:
    st.subheader("Marketing Source")
    original_source = st.selectbox("Original Source", ['PAID_SOCIAL', 'DIRECT_TRAFFIC', 'ORGANIC_SEARCH',
                                                        'PAID_SEARCH', 'REFERRALS', 'EMAIL_MARKETING',
                                                        'SOCIAL_MEDIA', 'AI_REFERRALS', 'OTHER_CAMPAIGNS', 'Unknown'])
    latest_source = st.selectbox("Latest Source", ['PAID_SOCIAL', 'DIRECT_TRAFFIC', 'ORGANIC_SEARCH',
                                                    'PAID_SEARCH', 'REFERRALS', 'EMAIL_MARKETING',
                                                    'SOCIAL_MEDIA', 'AI_REFERRALS', 'OTHER_CAMPAIGNS', 'Unknown'])
    device = st.selectbox("Device", ['Mobile', 'Desktop', 'Tablet', 'Unknown'])
    lead_type = st.selectbox("Lead Type", ['New enquiry', 'Repeat enquiry', 'Referral', 'Repeat traveller',
                                            'Travel agent', 'G2A internal', 'Unknown'])
    lead_source_detail = st.selectbox("Lead Source Detail", ['Online Ads', 'Google or other search', 'Social Media',
                                                              'From a friend', 'ChatGPT or other AI', 'Blog',
                                                              'Newsletter', 'Newspaper / Magazine', 'Unknown'])
    booking_intent = st.selectbox("Booking Intent", ['Help me research', 'Help me plan, so I can book later',
                                                      'Help me compare prices', 'Help me book my vacation', 'Unknown'])
    visitor_type = st.selectbox("Visitor Type", ['FIRST_VISITOR_NO_RESEARCH', 'FIRST_VISITOR_RESEARCH',
                                                  'PREVIOUS_VISITOR_ONCE', 'PREVIOUS_VISITOR_MULTIPLE', 'Unknown'])

with col3:
    st.subheader("Engagement Signals")
    enquiry_touchpoints = st.slider("Enquiry Touchpoints", 0, 25, 1)
    page_interactions = st.slider("Page Interactions", 0, 25, 1)
    unknown_engagement_1 = st.slider("Form Engagement", 0, 10, 1)
    total_visits = st.slider("Total Visits", 1, 30, 1)
    total_sessions = st.slider("Total Sessions", 1, 10, 1)
    average_pageviews = st.slider("Avg Pageviews", 1, 20, 2)
    number_of_sessions = st.slider("Number of Sessions", 1, 100, 1)
    time_on_site = st.slider("Time on Site (seconds)", 0, 1500, 30)
    returning_visitor = st.selectbox("Returning Visitor", [0, 1])
    engagement_score = st.slider("Engagement Score", 0, 5, 0)

st.divider()

if st.button("🔍 Score This Lead", use_container_width=True, type="primary"):

    input_data = {
        'marketing_original_source': original_source,
        'marketing_latest_source': latest_source,
        'device': device,
        'country': country,
        'budget_range': budget_range,
        'travel_group_type': travel_group,
        'trip_duration': trip_duration,
        'budget_level': budget_level,
        'currency': currency,
        'age_bracket': age_bracket,
        'lead_type': lead_type,
        'booking_intent': booking_intent,
        'enquiry_touchpoints': enquiry_touchpoints,
        'page_interactions': page_interactions,
        'unknown_engagement_1': unknown_engagement_1,
        'total_visits': total_visits,
        'total_sessions': total_sessions,
        'average_pageviews': average_pageviews,
        'number_of_sessions': number_of_sessions,
        'time_on_site': time_on_site,
        'returning_visitor': returning_visitor,
        'visitor_type': visitor_type,
        'lead_source_detail': lead_source_detail,
        'engagement_score': engagement_score,
        'has_specific_dates': 0,
        'sales_activity_count': 0,
        'contact_count': 0,
        'offer_count': 0,
        'response_indicator': 0,
        'follow_up_count': 0,
        'pipeline_stage_num': 0,
        'total_activities': 0,
    }

    for col, le in encoders.items():
        if col in input_data:
            val = input_data[col]
            if val in le.classes_:
                input_data[col] = le.transform([val])[0]
            else:
                if 'Unknown' in le.classes_:
                    input_data[col] = le.transform(['Unknown'])[0]
                else:
                    input_data[col] = 0

    input_df = pd.DataFrame([input_data])[feature_names]

    raw_conv = model_conv.predict(input_df)[0]
    profit = model_profit.predict(input_df)[0]

    raw_conv = max(0, min(raw_conv, 1.0))
    profit = max(0, profit)

    lead_score = raw_to_score(raw_conv)

    if lead_score >= 80:
        label = "TOP 🔥"
        css_class = "top"
        action = "Highest priority — senior rep calls within 2 hours"
        color = "#c62828"
        priority = "1"
    elif lead_score >= 50:
        label = "HIGH ⚡"
        css_class = "high"
        action = "Sales rep follows up within 24 hours with relevant options"
        color = "#e65100"
        priority = "2"
    elif lead_score >= 20:
        label = "MEDIUM 📧"
        css_class = "medium"
        action = "Add to targeted email campaign, check back in a week"
        color = "#2e7d32"
        priority = "3"
    else:
        label = "LOW ❄️"
        css_class = "low"
        action = "Add to general newsletter, no direct sales effort"
        color = "#1565c0"
        priority = "4"

    expected_value = raw_conv * profit

    st.divider()

    r1, r2, r3, r4 = st.columns(4)

    with r1:
        st.markdown(f"""
        <div class="score-box {css_class}">
            <h1 style="color: {color}; margin: 0;">{label}</h1>
            <p style="margin: 0;">Lead Score (Priority {priority})</p>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        st.metric("Lead Score", f"{lead_score:.0f} / 100")

    with r3:
        st.metric("Predicted Profit", f"${profit:,.0f}")

    with r4:
        st.metric("Expected Value", f"${expected_value:,.0f}")

    st.info(f"**Recommended Action:** {action}")

    st.divider()
    st.subheader("Score Breakdown")

    bcol1, bcol2 = st.columns(2)

    with bcol1:
        st.markdown("**Lead Score Tiers:**")
        tier_data = {
            'Tier': ['TOP 🔥', 'HIGH ⚡', 'MEDIUM 📧', 'LOW ❄️'],
            'Score Range': ['80 – 100', '50 – 79', '20 – 49', '0 – 19'],
            'Action': [
                'Senior rep calls within 2 hours',
                'Sales rep follows up within 24 hours',
                'Targeted email campaign',
                'General newsletter only'
            ]
        }
        st.table(pd.DataFrame(tier_data))

    with bcol2:
        st.markdown("**This Lead's Key Signals:**")
        signals = []
        if time_on_site > 300:
            signals.append(f"✅ High time on site ({time_on_site}s)")
        elif time_on_site < 60:
            signals.append(f"⚠️ Low time on site ({time_on_site}s)")

        if total_visits > 3:
            signals.append(f"✅ Multiple visits ({total_visits})")
        else:
            signals.append(f"⚠️ Few visits ({total_visits})")

        if enquiry_touchpoints > 2:
            signals.append(f"✅ Multiple enquiries ({enquiry_touchpoints})")
        else:
            signals.append(f"⚠️ Few enquiries ({enquiry_touchpoints})")

        if engagement_score >= 2:
            signals.append(f"✅ Good engagement score ({engagement_score})")
        else:
            signals.append(f"⚠️ Low engagement score ({engagement_score})")

        if returning_visitor == 1:
            signals.append("✅ Returning visitor")
        else:
            signals.append("⚠️ First-time visitor")

        if booking_intent in ['Help me book my vacation', 'Help me compare prices']:
            signals.append(f"✅ Strong intent: {booking_intent}")
        elif booking_intent == 'Help me research':
            signals.append(f"⚠️ Early stage: {booking_intent}")

        for s in signals:
            st.markdown(s)

        st.markdown(f"\n**Raw model probability:** {raw_conv:.4f} ({raw_conv*100:.2f}%)")
