import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import xgboost as xgb # <-- Nouvel import
import numpy as np
import json
import urllib.request

st.set_page_config(page_title="Dashboard Interactif Électromobilité", layout="wide")
st.title("Tableau de Bord Interactif : Électromobilité en France")

# =========================================================
# CHARGEMENT DES DONNÉES EN CACHE
# =========================================================
@st.cache_data
def load_data():
    # 1. VE (Véhicules)
    df_ve = pd.read_parquet('vehicules_electriques.parquet')
    
    # 2. Typologie
    try:
        df_typo = pd.read_csv('nombre-de-points-de-charge-par-typologie.csv', encoding='utf-8')
    except:
        df_typo = pd.read_csv('nombre-de-points-de-charge-par-typologie.csv', encoding='latin1')

    # 3. Bornes
    try:
        df_pc = pd.read_csv('bornes-irve (1).csv', sep=';', low_memory=False)
    except:
        df_pc = pd.DataFrame()

    # 4. Ratio par département
    try:
        df_ratio = pd.read_csv('ratio_par_departement.csv')
    except:
        df_ratio = pd.DataFrame()

    return df_ve, df_typo, df_pc, df_ratio

@st.cache_data
def load_geojson():
    url_geojson = "https://github.com/gregoiredavid/france-geojson/raw/master/departements-version-simplifiee.geojson"
    with urllib.request.urlopen(url_geojson) as url:
        return json.load(url)

with st.spinner("Chargement des données..."):
    df_ve, df_typo, df_pc, df_ratio = load_data()
    departements_geo = load_geojson()

# Création des onglets
tab1, tab2, tab3, tab4 = st.tabs([
    "Tendance Flotte VE", 
    "Typologie des Bornes", 
    "Diagnostic IA (XGBoost)", # <-- Nom mis à jour
    "Carte Animée Ratio"
])

# =========================================================
# ONGLET 1 : TENDANCE FLOTTE VE
# =========================================================
with tab1:
    st.header("Évolution de la flotte de Véhicules Électriques")
    
    all_depts = sorted(df_ve['CODGEO'].astype(str).str[:2].unique())
    selected_depts = st.multiselect("Filtrer par Département (laissez vide pour la France entière)", all_depts)
    
    df_ve_filtered = df_ve.copy()
    if selected_depts:
        df_ve_filtered = df_ve_filtered[df_ve_filtered['CODGEO'].astype(str).str[:2].isin(selected_depts)]

    if not df_ve_filtered.empty:
        df_ve_filtered['DATE_ARRETE'] = pd.to_datetime(df_ve_filtered['DATE_ARRETE'])
        df_VE_agg = df_ve_filtered.groupby('DATE_ARRETE')['NB_VP_RECHARGEABLES_EL'].sum().reset_index().sort_values('DATE_ARRETE')
        df_VE_agg['days'] = (df_VE_agg['DATE_ARRETE'] - df_VE_agg['DATE_ARRETE'].min()).dt.days

        X = df_VE_agg[['days']]
        y = df_VE_agg['NB_VP_RECHARGEABLES_EL']
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)

        jours_futurs = st.slider("Jours de prédiction future :", 0, 365, 90)
        if jours_futurs > 0:
            future_days = np.arange(X['days'].max() + 1, X['days'].max() + 1 + jours_futurs).reshape(-1, 1)
            future_pred = model.predict(future_days)
            future_dates = pd.date_range(start=df_VE_agg['DATE_ARRETE'].max() + pd.Timedelta(days=1), periods=jours_futurs)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=df_VE_agg['DATE_ARRETE'], y=y, mode='markers', name='Données réelles', marker=dict(color='teal')))
        fig1.add_trace(go.Scatter(x=df_VE_agg['DATE_ARRETE'], y=y_pred, mode='lines', name='Tendance actuelle', line=dict(color='red', dash='dash')))
        
        if jours_futurs > 0:
            fig1.add_trace(go.Scatter(x=future_dates, y=future_pred, mode='lines', name='Prédiction', line=dict(color='orange', dash='dot')))

        fig1.update_layout(title="Évolution et Prédiction des VE", xaxis_title="Date", yaxis_title="Nombre de VE", hovermode="x unified")
        st.plotly_chart(fig1, use_container_width=True)

# =========================================================
# ONGLET 2 : TYPOLOGIE DES BORNES
# =========================================================
with tab2:
    st.header("Répartition des points de charge par type")
    
    if not df_typo.empty:
        df_t = df_typo.copy()
        if df_t.shape[1] == 1:
            df_t = df_t.iloc[:, 0].str.split('[,;]', expand=True)
        df_t = df_t.iloc[:, :5]
        df_t.columns = ['Annee', 'Trimestre', 'Mois_Label', 'Typologie', 'Nombre']
        df_t['Nombre'] = pd.to_numeric(df_t['Nombre'].astype(str).str.replace('"', '').str.strip(), errors='coerce')
        df_t = df_t.dropna(subset=['Nombre'])

        view_mode = st.radio("Mode d'affichage", ["Cumulé (Empilé)", "Comparatif (Côte à côte)", "Proportion (100%)"])
        barmode = 'stack' if view_mode == "Cumulé (Empilé)" else 'group' if view_mode == "Comparatif (Côte à côte)" else 'relative'
        
        fig2 = px.bar(df_t.sort_values(['Annee', 'Trimestre']), x='Mois_Label', y='Nombre', color='Typologie', 
                      barmode=barmode, title="Évolution de la Typologie",
                      color_discrete_sequence=px.colors.qualitative.Set2)
        
        if view_mode == "Proportion (100%)":
            fig2.update_layout(barnorm='percent', yaxis_title="% du total")
            
        st.plotly_chart(fig2, use_container_width=True)

# =========================================================
# ONGLET 3 : DIAGNOSTIC IA (XGBoost)
# =========================================================
with tab3:
    st.header("Diagnostic Territorial IA (XGBoost)")
    st.markdown("Comparaison entre le parc de Véhicules Électriques et l'infrastructure de recharge.")
    
    if not df_ve.empty and not df_pc.empty:
        # 1. Préparation des données (Identique à ton script)
        def get_dept(c):
            if pd.isna(c): return 'UNK'
            c = str(c)
            return c[:3] if c.startswith('97') else c[:2]

        df_ve_ia = df_ve.copy()
        df_ve_ia['dept'] = df_ve_ia['CODGEO'].apply(get_dept)
        df_ve_agg = df_ve_ia.groupby(['dept', 'DATE_ARRETE'])['NB_VP_RECHARGEABLES_EL'].sum().reset_index()

        df_pc_ia = df_pc.copy()
        df_pc_ia['date_maj'] = pd.to_datetime(df_pc_ia['date_maj'], errors='coerce')
        df_pc_ia = df_pc_ia.dropna(subset=['date_maj'])
        geo_col = 'code_insee_commune' if 'code_insee_commune' in df_pc_ia.columns else 'code_insee'
        df_pc_ia['dept'] = df_pc_ia[geo_col].astype(str).apply(get_dept)

        dates_cles = sorted(df_ve_agg['DATE_ARRETE'].unique())
        snapshots = []
        for d in dates_cles:
            mask = df_pc_ia['date_maj'] <= d
            bornes_a_date = df_pc_ia[mask]
            if 'nbre_pdc' in bornes_a_date.columns:
                stock_dept = bornes_a_date.groupby('dept')['nbre_pdc'].sum().reset_index()
            else:
                stock_dept = bornes_a_date.groupby('dept').size().reset_index(name='nbre_pdc')
            stock_dept['DATE_ARRETE'] = d
            snapshots.append(stock_dept)

        df_pc_history = pd.concat(snapshots, ignore_index=True)
        df = pd.merge(df_ve_agg, df_pc_history, on=['dept', 'DATE_ARRETE'], how='inner')
        df.rename(columns={'nbre_pdc': 'nb_pdc'}, inplace=True)
        df['nb_pdc'] = df['nb_pdc'].fillna(0)
        df = df[df['dept'] != '00']

        # 2. Modèle XGBoost
        last_date = df['DATE_ARRETE'].max()
        df_analyse = df[df['DATE_ARRETE'] == last_date].copy()

        # Entraînement SANS Paris (75)
        df_train = df_analyse[df_analyse['dept'] != '75'].copy()
        X_train = df_train[['NB_VP_RECHARGEABLES_EL']]
        y_train = df_train['nb_pdc']

        X_all = df_analyse[['NB_VP_RECHARGEABLES_EL']]
        y_all = df_analyse['nb_pdc']

        xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, random_state=42)
        xgb_model.fit(X_train, y_train)

        df_analyse['pred_pdc_xgb'] = xgb_model.predict(X_all)
        df_analyse['ecart_xgb'] = df_analyse['nb_pdc'] - df_analyse['pred_pdc_xgb']

        # 3. Graphique Plotly interactif
        fig3 = px.scatter(df_analyse, x='NB_VP_RECHARGEABLES_EL', y='nb_pdc', 
                          color='ecart_xgb', size=abs(df_analyse['ecart_xgb']),
                          hover_name='dept',
                          hover_data={'ecart_xgb': ':.0f', 'pred_pdc_xgb': ':.0f'},
                          color_continuous_scale='RdYlGn',
                          labels={'NB_VP_RECHARGEABLES_EL': 'Parc de Voitures Électriques', 
                                  'nb_pdc': 'Nb Bornes Réelles', 
                                  'ecart_xgb': 'Écart vs Modèle'},
                          title=f"Diagnostic Territorial IA (XGBoost) au {last_date.date()}")
        
        # Ajout de la courbe "Norme Nationale" (Ligne Bleue)
        x_virtuel = pd.DataFrame({'NB_VP_RECHARGEABLES_EL': np.linspace(df_analyse['NB_VP_RECHARGEABLES_EL'].min(), df_analyse['NB_VP_RECHARGEABLES_EL'].max(), 500)})
        y_virtuel = xgb_model.predict(x_virtuel)
        
        fig3.add_trace(go.Scatter(x=x_virtuel['NB_VP_RECHARGEABLES_EL'], y=y_virtuel, 
                                  mode='lines', name='Norme Nationale', 
                                  line=dict(color='blue', width=3)))

        # Limitation de l'axe Y pour ne pas écraser à cause de Paris
        ymax = max(df_analyse[df_analyse['dept'] != '75']['nb_pdc'].max() * 1.5, 5000)
        fig3.update_layout(yaxis_range=[-500, ymax])
        
        st.plotly_chart(fig3, use_container_width=True)

        # 4. Affichage des Top 5 en Dataframes
        st.divider()
        col1, col2 = st.columns(2)
        
        # Mise en forme pour l'affichage
        df_display = df_analyse[['dept', 'NB_VP_RECHARGEABLES_EL', 'nb_pdc', 'pred_pdc_xgb', 'ecart_xgb']].copy()
        df_display['pred_pdc_xgb'] = df_display['pred_pdc_xgb'].astype(int)
        df_display['ecart_xgb'] = df_display['ecart_xgb'].astype(int)
        df_display.columns = ['Département', 'Nb Véhicules', 'Nb Bornes', 'Prédiction IA', 'Écart']

        with col1:
            st.subheader("🟢 Top 5 : Départements les plus surdotés")
            top_avance = df_display.sort_values('Écart', ascending=False).head(5)
            st.dataframe(top_avance, hide_index=True, use_container_width=True)

        with col2:
            st.subheader("🔴 Top 5 : Départements les plus sous-dotés")
            top_retard = df_display.sort_values('Écart').head(5)
            st.dataframe(top_retard, hide_index=True, use_container_width=True)

# =========================================================
# ONGLET 4 : CARTE ANIMÉE RATIO
# =========================================================
with tab4:
    st.header("Évolution du Ratio : Nombre de VE pour 1 Point de Charge")
    
    if df_ratio.empty and not df_ve.empty and not df_pc.empty:
        df_ratio = df.copy()
        df_ratio['ratio'] = np.where(df_ratio['nb_pdc'] > 0, df_ratio['NB_VP_RECHARGEABLES_EL'] / df_ratio['nb_pdc'], 0)
        df_ratio['date'] = df_ratio['DATE_ARRETE'].astype(str)

    if not df_ratio.empty:
        df_ratio['dept'] = df_ratio['dept'].apply(lambda x: str(x).zfill(2) if len(str(x)) < 3 else str(x))
        if 'date_str' not in df_ratio.columns:
            df_ratio['date_str'] = df_ratio['date'].astype(str).str[:10]

        fig4 = px.choropleth(
            df_ratio.sort_values('date'), 
            geojson=departements_geo, 
            locations='dept', 
            featureidkey="properties.code", 
            color='ratio', 
            animation_frame='date_str', 
            color_continuous_scale="RdYlGn_r", 
            range_color=[0, 30], 
            title="Évolution du Ratio (VE / Borne) par département",
            labels={'ratio': 'Ratio (VE / Borne)', 'date_str': 'Date'}
        )

        fig4.update_geos(fitbounds="locations", visible=False)
        fig4.update_layout(margin={"r":0,"t":50,"l":0,"b":0})

        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.warning("Impossible de générer la carte : Données manquantes.")
