import streamlit as st
from PIL import Image
import os
import calendar
from joblib import load
# import pyLDAvis
# import pyLDAvis.gensim_models as gensimvis
import streamlit.components.v1 as components

st.set_page_config(
    page_title="Employee Behaviour Analysis",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

image_directory = 'Plots'
image_files = [os.path.join(image_directory, filename) for filename in os.listdir(image_directory) if filename.endswith(".png")]

image_files.sort()

##########################################################################################################################


st.title('Number of Visits to Websites')
# st.write('Below is the plot for showing login and logoff times by profession. Use the dropdown to select the profession.')

image = Image.open("./Plots/Number of Visits to Websites.png")
st.image(image)


##########################################################################################################################

st.title('Heatmap of Upload Events for Top 15 Users')

image = Image.open("./Plots/Heatmap.png")
st.image(image)

##########################################################################################################################

st.title('Top Frequent Sequences')

image = Image.open("./Plots/Top Frequent Seq All Year.png")
st.image(image)

image = Image.open("./Plots/Top Frequent Seq.png")
st.image(image)

###########################################################################################################################

st.title('Login Average by Profession')
st.write('Below is the plot for showing login average by month. Use the dropdown to select the month.')

month = []*12
month_str = []*12

for plt in image_files:
    if("Login Avg by Profession " in plt):
        month.append(plt)
        month_str.append(plt.split(' ')[-1])
     
month_str = [i.split(".")[0] for i in month_str]
month_str = sorted(month_str, key=lambda x: list(calendar.month_name).index(x))

selected_mon1 = st.selectbox("Select Month", month_str, key=1)
image = Image.open("./Plots/Login Avg by Profession {}.png".format(selected_mon1))
st.image(image)

############################################################################################################################

st.title('Average Uniqe Machine Usage by Month')
st.write('Below is the plot for showing login Average Uniqe Machine Usage by Month. Use the dropdown to select the month.')


month = []*12
month_str = []*12

for plt in image_files:
    if("Avg Uniq Machine Usage for " in plt):
        month.append(plt)
        month_str.append(plt.split(' ')[-1])

month_str = [i.split(".")[0] for i in month_str]
month_str = sorted(month_str, key=lambda x: list(calendar.month_name).index(x))

selected_mon2 = st.selectbox("Select Month", month_str, key=2)
image = Image.open("./Plots/Avg Uniq Machine Usage for {}.png".format(selected_mon2))
st.image(image)

###########################################################################################################################

st.title('Login and Logoff by Profession')
st.write('Below is the plot for showing login and logoff times by profession. Use the dropdown to select the profession.')


login_and_off = []*42
login_and_off_str = []*42

for plt in image_files:
    if("LoginAndOff" in plt):
        login_and_off.append(plt)
        login_and_off_str.append(plt.split('-')[1])
     
login_and_off_str = [i.split(".")[0] for i in login_and_off_str]

selected_log = st.selectbox("Select Profession", login_and_off_str)
image = Image.open("./Plots/LoginAndOff-{}.png".format(selected_log))
st.image(image)

##########################################################################################################################

st.title('LDA Topic Model Visualization')

# def load_models():
#     dictionary = load('dictionary.joblib')
#     corpus = load('corpus.joblib')
#     lda_model = load('lda_model.joblib')
#     return dictionary, corpus, lda_model

# dictionary, corpus, lda_model = load_models()
# lda_display = gensimvis.prepare(lda_model, corpus, dictionary, sort_topics=False)

# pyLDAvis.save_html(lda_display, 'lda.html')

HtmlFile = open("./lda.html", 'r', encoding='utf-8')
source_code = HtmlFile.read() 
print(source_code)
components.html(source_code, height=800, width=1250, scrolling=False)

##########################################################################################################################

st.title('Topic distribution for Technical Trainers the entire Year')

def load_models_bert():
    sentence_model = load('sentence_model.joblib')
    topic_model = load('topic_model.joblib')
    embeddings = load('embeddings.joblib')
    documents = load('documents.joblib')
    return sentence_model, topic_model, embeddings, documents

sentence_model, topic_model, embeddings, documents = load_models_bert()

def visualize_documents(topic_model, embeddings):
    fig = topic_model.visualize_documents(documents, embeddings=embeddings, title="Topic distribution for Technical Trainers the entire Year")
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(
            color="black",
            size=12
        ),
        width=1000,
        height=600,
        legend=dict(
            bgcolor='white',
            font=dict(
                color='black'
            )
        )
    )

    fig.update_xaxes(
        tickfont=dict(color='black'), 
        title_font=dict(color='black')
    )
    fig.update_yaxes(
        tickfont=dict(color='black'), 
        title_font=dict(color='black')
    )
    
    fig.update_xaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    fig.update_yaxes(showline=True, linewidth=2, linecolor='black', gridcolor='gray')
    
    st.plotly_chart(fig, use_container_width=True)

visualize_documents(topic_model, embeddings)
