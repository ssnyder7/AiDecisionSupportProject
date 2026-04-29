import streamlit as st
import pandas as pd
import steamreviews
from sentence_transformers import SentenceTransformer, util

# APP CONFIG & MODELS
st.set_page_config(page_title="Steam Review Analyzer", page_icon="🎮")
model = SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def load_data():
    return pd.read_excel("TitleToAppID.csv", engine='openpyxl')

df = load_data()

# ANALYSIS ENGINE
def get_game_data(clean_desc, app_id):
    # Fetch reviews
    steam_review_set, _ = steamreviews.download_reviews_for_app_id(
        app_id, chosen_request_params={'language': 'english'}
    )
    
    summary = steam_review_set.get('query_summary', {})
    all_reviews_dict = steam_review_set.get('reviews', {})
    reviews_list = [v.get('review') for v in all_reviews_dict.values()]

    if not reviews_list:
        return None

    # Optimized Batch Encoding
    desc_embedding = model.encode(clean_desc, convert_to_tensor=True)
    review_embeddings = model.encode(reviews_list, convert_to_tensor=True)
    cosine_scores = util.cos_sim(desc_embedding, review_embeddings)
    match_score = float(cosine_scores.mean()) * 100
    
    return {
        "score_desc": summary.get('review_score_desc', "No score"),
        "pos_pct": (summary.get('total_positive', 0) / summary.get('total_reviews', 1)) * 100,
        "total": summary.get('total_reviews', 0),
        "match": match_score
    }

# UI LAYOUT
st.title("🎮 Steam Marketing vs. Reality")
game_input = st.text_input("Enter a game name:", placeholder="e.g. Sonic Mania (Warning: popular games may take quite a long time)")

if game_input:
    match = df[df['Name'].str.contains(game_input, case=False, na=False)]
    
    if not match.empty:
        game = match.iloc[0]
        st.subheader(f"Analyzing: {game['Name']}")
        
        with st.spinner("Fetching community reviews..."):
            info = get_game_data(game['GameDescription'], game['app_id'])
        
        if info:
            # Calculate color: 0 is dark red (139,0,0), 50+ is bright green (0,255,0)
            # We'll clamp the match score to 50 for the max green intensity
            score_clamped = min(max(info['match'], 0), 50)
            ratio = score_clamped / 50
            
            r = int(139 * (1 - ratio))
            g = int(255 * ratio)
            b = 0
            color_hex = f'#{r:02x}{g:02x}{b:02x}'

            # Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Community Rating", info['score_desc'])
            col2.metric("Positive Reviews", f"{info['pos_pct']:.1f}%")
            col3.metric("Total Reviews", f"{info['total']:,}")

            st.markdown(f"""
                <div style="background-color:#1e1e1e; padding:20px; border-radius:10px; border-left: 10px solid {color_hex};">
                    <h3 style="margin:0;">Marketing Alignment Score: <span style="color:{color_hex};">{info['match']:.2f}%</span></h3>
                    <p style="color:#888; font-size:0.9em;">This score represents how closely the Steam store description matches actual player feedback using semantic similarity (Scores range from 0%-50%).</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.error("Game not found in local database. Try a different title.")
    else:
        st.error("Game not found in your CSV.")
