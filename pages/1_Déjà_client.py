import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.pyfunc
import shap
from lightgbm import LGBMClassifier
import plotly.graph_objects as go
import joblib

#############
# Affichage #
#############
st.set_page_config(page_title="Déjà Client", page_icon="👑")
url_image = "https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png"
st.image(url_image, use_column_width=True)
st.write("# Le client est déjà connu de nos services")

# Ajout d'un slider pour modifier la taille des textes (WCAG)
st.markdown(f"<p style='font-size:22px;'><strong>Ajustez la taille du texte des graphiques :</strong></p>", unsafe_allow_html=True)
font_size = st.slider("", min_value=10, max_value=42, value=18)
title_buff = 14

#######################
# Liens et chargement #
#######################
model_path = "mlflow_model"

@st.cache_resource
def load_all():
    model = mlflow.lightgbm.load_model(model_path)
    df_brut = pd.read_csv("Dataframe_brut.csv")
    scaler = joblib.load("scaler.pkl")
    return model, df_brut, scaler

model, df_brut, scaler = load_all()

###################################
# Fonction graphique jauge défaut #
###################################
def plot_gauge(probability, cutoff, title_font_size=font_size):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100, 
        title={'text': "Probabilité de Défaut (%)", 'font': {'size': title_font_size}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "grey"},
            'steps': [
                {'range': [0, cutoff * 100], 'color': "lightgreen"},
                {'range': [cutoff * 100, 100], 'color': "orangered"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': cutoff * 100
            }
        }
    ))
    return fig

# Initialiser les états de variables
if "modified" not in st.session_state:
    st.session_state.modified = False
    st.session_state.modified_client_data = None

if "show_form" not in st.session_state:
    st.session_state.show_form = False

if "selected_variable" not in st.session_state:
    st.session_state.selected_variable = None

####################
# Début du process #
####################
st.markdown(f"<p style='font-size:{font_size+title_buff}px;'><strong>Entrez l'identifiant client (SK_ID_CURR) :</strong></p>", unsafe_allow_html=True)
sk_id_curr = st.text_input("", "")

if sk_id_curr:
    try:
        sk_id_curr = int(sk_id_curr)
    except ValueError:
        sk_id_curr = -1

    # Charger les données du client
    client_data = df_brut[df_brut["SK_ID_CURR"] == sk_id_curr].drop(columns=['TARGET']).reset_index(drop=True)
    #client_data_norm = df_norm[df_norm["SK_ID_CURR"] == sk_id_curr].drop(columns=['TARGET']).reset_index(drop=True)
    client_data_display = client_data.set_index('SK_ID_CURR')

    if client_data.empty:
        st.error("Aucun client trouvé avec cet identifiant.")
    else:
        # Afficher les données clients
        st.markdown(f"<p style='font-size:{font_size}px;'>Analyse des données client :</p>", unsafe_allow_html=True)
        st.write(client_data_display)

        ########################
        # Prédiction du modèle #
        ########################
        try:
            # Normalisation des données
            data_to_scale = client_data.drop(columns=['SK_ID_CURR'], errors='ignore')
            client_data_norm = pd.DataFrame(scaler.transform(data_to_scale), columns=data_to_scale.columns)
            # Réintégrer SK_ID_CURR
            client_data_norm['SK_ID_CURR'] = client_data['SK_ID_CURR'].values
            # Prédiction et affichage
            prediction = model.predict(client_data_norm)
            probability = model.predict_proba(client_data_norm)[:, 1]
            probability_value = probability[0]

            # Jauge de probabilité
            cutoff = 0.089
            st.plotly_chart(plot_gauge(probability[0], cutoff))
            st.markdown(f"<div style='font-size:{font_size+8}px;'>La probabilité de défaut du client est de : <span style='color:blue;'>{probability_value * 100:.6f}%</span></div>", unsafe_allow_html=True)

            st.markdown(f"<div style='font-size:{font_size+8}px;'>Le seuil d'acceptation est de : <span style='color:red;'>{cutoff * 100:.2f}%</span></div>", unsafe_allow_html=True)
            
            if probability[0] > cutoff:
                st.markdown(f"<div style='font-size:{font_size+8}px;'>Cette probabilité est <strong>supérieur</strong> au seuil, la demande de prêt du client doit être :</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:red;font-size:{font_size+title_buff}px;'>Refusé</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='font-size:{font_size+8}px;'>Cette probabilité est <strong>inférieur</strong> au seuil, la demande de prêt du client peut être : ", unsafe_allow_html=True)
                st.markdown(f"<div style='color:green;font-size:{font_size+title_buff}px;'><strong>Accepté</strong></div>", unsafe_allow_html=True)
            st.write("")

            ########################################
            # Représentation graphique des données #
            ########################################
            st.markdown(f"<div style='font-size:{font_size+title_buff}px;'><strong>Analyse graphique :</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:{font_size}px;'>Sélectionnez une variable à examiner :</div>", unsafe_allow_html=True)
            selected_variable = st.selectbox("", client_data.drop(columns=['SK_ID_CURR']).columns)
            mean_value = df_brut[selected_variable].mean()
            median_value = df_brut[selected_variable].median()
            client_value = client_data[selected_variable].iloc[0]

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))

            # Distribution de la population
            ax.hist(df_brut[selected_variable].dropna(), bins=30, color='gray', alpha=0.7, label='Population')

            # Ajouter des repères pour le client, la moyenne et la médiane
            ax.axvline(client_value, color='blue', linestyle='--', linewidth=3, label=f'Client ({client_value:.2f})')
            ax.axvline(mean_value, color='green', linestyle='-', linewidth=3, label=f'Moyenne ({mean_value:.2f})')
            ax.axvline(median_value, color='orange', linestyle='dotted', linewidth=3, label=f'Médiane ({median_value:.2f})')

            # Ajouter des annotations textuelles
            ax.text(client_value, ax.get_ylim()[1] * 0.8, "Valeur client", color="blue", fontsize=font_size)
            ax.text(mean_value, ax.get_ylim()[1] * 0.6, "Moyenne", color="green", fontsize=font_size)

            # Légendes et titres
            ax.set_xlabel(selected_variable, fontsize=font_size)
            ax.set_ylabel('Fréquence', fontsize=font_size)
            ax.set_title(f"Distribution de la variable '{selected_variable}'", fontsize=font_size)
            ax.legend()

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
            st.caption(f"Ce graphique montre la distribution de la variable {selected_variable}, avec des repères pour le client, la moyenne et la médiane.")

        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

        ######################
        # Feature Importance #
        ######################
        try:
            st.markdown(f"<div style='font-size:{font_size+title_buff}px;'><strong>Importance des variables du client pour la décision du modèle</strong></div>", unsafe_allow_html=True)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(client_data_norm)
            shap_values_single = shap_values[1] if isinstance(shap_values, list) else shap_values
            base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            # Filtrer pour exclure SK_ID_CURR des données et des SHAP values
            filtered_data = client_data_norm.drop(columns=['SK_ID_CURR'], errors='ignore')
            filtered_shap_values = shap_values_single[:, filtered_data.columns.get_indexer(filtered_data.columns)]
            # Graphique
            plt.figure()
            shap.waterfall_plot(shap.Explanation(values=filtered_shap_values[0], 
                                                 base_values=base_value, 
                                                 data=filtered_data.iloc[0]))
            st.pyplot(plt.gcf())
            st.caption("""
            Ce graphique montre l'impact des différentes variables sur la décision du modèle pour ce client. Chaque barre représente une variable :
            - **Bleu** : variables ayant contribué à réduire le risque de défaut. (flèches vers la gauche)
            - **Rouge** : variables ayant contribué à augmenter le risque de défaut. (flèches vers la droite)
            
            La longueur des barres indique l'importance de chaque variable dans la décision. Par exemple, une variable avec une grande barre a fortement influencé le modèle. Les valeurs numériques indiquent les données spécifiques de ce client pour chaque variable.
            """)
        except Exception as e:
            st.error(f"Erreur lors de la feature importance : {e}")

        #####################################
        # Modifications des données clients #
        #####################################
        if st.button("Modifier les données client"):
            st.session_state.show_form = True
        
        # Afficher le formulaire uniquement si le drapeau est actif
        if st.session_state.show_form:
            st.markdown(f"<div style='font-size:{font_size+title_buff}px;'><strong>Modifier une ou plusieurs variables pour ce client :</strong></div>", unsafe_allow_html=True)

            # Créer une copie modifiable des données du client
            modified_client_data = client_data.copy()

            # Formulaire de modification
            with st.form("modify_form"):
                for var in client_data.drop(columns=['SK_ID_CURR']).columns:
                    if np.issubdtype(client_data[var].dtype, np.number):
                        modified_client_data[var] = st.number_input(f"{var} :", value=float(client_data[var].iloc[0]))
                    else:
                        modified_client_data[var] = st.text_input(f"{var} :", value=str(client_data[var].iloc[0]))
                submitted = st.form_submit_button("Recalculer la prédiction")

                if submitted:
                    st.session_state.modified = True
                    st.session_state.modified_client_data = modified_client_data
                    
                    # Affichage des données modifiées
                    modified_client_data_display = modified_client_data.set_index('SK_ID_CURR')
                    st.markdown(f"<p style='font-size:{font_size}px;'>Données client modifiées :</p>", unsafe_allow_html=True)
                    st.write(modified_client_data_display)
                    # Supprimer SK_ID_CURR pour la normalisation
                    data_to_scale = modified_client_data.drop(columns=['SK_ID_CURR'], errors='ignore')
                    modified_client_data_norm = pd.DataFrame(scaler.transform(data_to_scale), columns=data_to_scale.columns)
                    # Réintégrer SK_ID_CURR
                    modified_client_data_norm['SK_ID_CURR'] = modified_client_data['SK_ID_CURR'].values

                    #######################
                    # Nouvelle prédiction #
                    #######################
                    prediction = model.predict(modified_client_data_norm)
                    probability = model.predict_proba(modified_client_data_norm)[:, 1]
                    probability_value = probability[0]

                    # Jauge de probabilité
                    cutoff = 0.089
                    st.plotly_chart(plot_gauge(probability[0], cutoff))
                    st.markdown(f"<div style='font-size:{font_size+8}px;'>La probabilité de défaut du client est de : <span style='color:blue;'>{probability_value * 100:.6f}%</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div style='font-size:{font_size+8}px;'>Le seuil d'acceptation est de : <span style='color:red;'>{cutoff * 100:.2f}%</span></div>", unsafe_allow_html=True)

                    if probability[0] > cutoff:
                        st.markdown(f"<div style='font-size:{font_size+8}px;'>Cette probabilité est <strong>supérieur</strong> au seuil, la demande de prêt du client doit être :</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='color:red;font-size:{font_size+title_buff}px;'>Refusé</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div style='font-size:{font_size+8}px;'>Cette probabilité est <strong>inférieur</strong> au seuil, la demande de prêt du client peut être : ", unsafe_allow_html=True)
                        st.markdown(f"<div style='color:green;font-size:{font_size+title_buff}px;'><strong>Accepté</strong></div>", unsafe_allow_html=True)
                        
                    ######################
                    # Feature Importance #
                    ######################
                    try:
                        st.markdown(f"<div style='font-size:{font_size+title_buff}px;'><strong>Importance des variables du client pour la décision du modèle</strong></div>", unsafe_allow_html=True)
                        explainer = shap.TreeExplainer(model)
                        shap_values = explainer.shap_values(modified_client_data_norm)
                        shap_values_single = shap_values[1] if isinstance(shap_values, list) else shap_values
                        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
                        # Filtrer pour exclure SK_ID_CURR des données et des SHAP values
                        filtered_data = modified_client_data_norm.drop(columns=['SK_ID_CURR'], errors='ignore')
                        filtered_shap_values = shap_values_single[:, filtered_data.columns.get_indexer(filtered_data.columns)]
                        # Graphique
                        plt.figure()
                        shap.waterfall_plot(shap.Explanation(values=filtered_shap_values[0], 
                                                             base_values=base_value, 
                                                             data=filtered_data.iloc[0]))
                        st.pyplot(plt.gcf())
                        st.caption("""
                        Ce graphique montre l'impact des différentes variables sur la décision du modèle pour ce client. Chaque barre représente une variable :
                        - **Bleu** : variables ayant contribué à réduire le risque de défaut. (flèches vers la gauche)
                        - **Rouge** : variables ayant contribué à augmenter le risque de défaut. (flèches vers la droite)
                        
                        La longueur des barres indique l'importance de chaque variable dans la décision. Par exemple, une variable avec une grande barre a fortement influencé le modèle. Les valeurs numériques indiquent les données spécifiques de ce client pour chaque variable.
                        """)
                    except Exception as e:
                        st.error(f"Erreur lors de la feature importance : {e}")

            ########################################
            # Représentation graphique des données #
            ########################################
            st.markdown(f"<div style='font-size:{font_size+title_buff}px;'><strong>Analyse graphique :</strong></div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:{font_size}px;'>Sélectionnez une variable à examiner :</div>", unsafe_allow_html=True)
            selected_variable_modif = st.selectbox(" ", modified_client_data.drop(columns=['SK_ID_CURR']).columns)
            mean_value = df_brut[selected_variable_modif].mean()
            median_value = df_brut[selected_variable_modif].median()
            client_value = modified_client_data[selected_variable_modif].iloc[0]

            # Création du graphique
            fig, ax = plt.subplots(figsize=(10, 6))

            # Distribution de la population
            ax.hist(df_brut[selected_variable_modif].dropna(), bins=30, color='gray', alpha=0.7, label='Population')

            # Ajouter des repères pour le client, la moyenne et la médiane
            ax.axvline(client_value, color='blue', linestyle='--', linewidth=3, label=f'Client ({client_value:.2f})')
            ax.axvline(mean_value, color='green', linestyle='-', linewidth=3, label=f'Moyenne ({mean_value:.2f})')
            ax.axvline(median_value, color='orange', linestyle='dotted', linewidth=3, label=f'Médiane ({median_value:.2f})')

            # Ajouter des annotations textuelles
            ax.text(client_value, ax.get_ylim()[1] * 0.8, "Valeur client", color="blue", fontsize=font_size)
            ax.text(mean_value, ax.get_ylim()[1] * 0.6, "Moyenne", color="green", fontsize=font_size)

            # Légendes et titres
            ax.set_xlabel(selected_variable_modif, fontsize=font_size)
            ax.set_ylabel('Fréquence', fontsize=font_size)
            ax.set_title(f"Distribution de la variable '{selected_variable_modif}'")
            ax.legend()

            # Afficher le graphique dans Streamlit
            st.pyplot(fig)
            st.caption(f"Ce graphique montre la distribution de la variable {selected_variable}, avec des repères pour le client, la moyenne et la médiane.")
