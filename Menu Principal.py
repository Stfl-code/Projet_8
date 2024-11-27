import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(page_title="Chargement des données", page_icon="⚙")

# URL de l'image
url_image = "https://user.oc-static.com/upload/2023/03/22/16794938722698_Data%20Scientist-P7-01-banner.png"
# Afficher l'image depuis l'URL
st.image(url_image, use_column_width=True)

# Titre
st.markdown("<h1 style='text-align: center;'>Menu principal</h1>", unsafe_allow_html=True)
st.markdown("<h2>Veuillez sélectionner une page via le menu déroulant ci-dessous ou le menu latéral.</h3>", unsafe_allow_html=True)

# Ajout d'un slider pour modifier la taille des textes (WCAG)
st.markdown(f"<p style='font-size:22px;'><strong>Ajustez la taille du texte des graphiques :</strong></p>", unsafe_allow_html=True)
font_size = st.slider("", min_value=10, max_value=42, value=18)

# Liste des pages disponibles (correspond au nom des fichiers sans les préfixes numériques)
pages = {
    "Déjà client": "Déjà client",
    "Nouveau client": "Nouveau client",
}

# Menu déroulant
st.markdown(f"<p style='font-size:{font_size}px;'><strong>Sélectionnez une page :</strong></p>", unsafe_allow_html=True)
selected_page = st.selectbox("", options=list(pages.keys()))

# Rediriger vers la page sélectionnée
if st.button("Aller à la page sélectionnée"):
    switch_page(pages[selected_page])
    
