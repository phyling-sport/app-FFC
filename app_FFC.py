import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from PIL import Image
import matplotlib.pyplot as plt
import plotly_express as px
import os
import streamlit as st
from streamlit_player import st_player
import datetime as dt
from datetime import datetime
from scipy import stats
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import s3fs


st.set_page_config(page_title='Phyling FFC')
# logo = Image.open('C:/Users/Chevallier/Phyling Dropbox/Phyling/Com/Logo/logo PHYLING_grand.png')

groupe = st.sidebar.selectbox(
    "Groupe",
    ("phyling","Sprint","Endurance")
)

choose = st.sidebar.selectbox(
    "Exercice",
    ("CMJ", "Nordic Harmstring", "Développé couché")
)

analyse= st.sidebar.selectbox(
    "Analyse",
    ("Suivi d'indicateurs","Statistiques","Comparaisons de courbes")
)


# Create connection object.
# `anon=False` means not anonymous, i.e. it uses access keys to pull data.
fs = s3fs.S3FileSystem(anon=False)

# Retrieve file contents.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def read_file(filename):
    with fs.open(filename) as f:
        return pd.read_csv(f,header=1,delimiter=';',decimal='.')#.read().decode("utf-8")
def read_file_courbes(filename):
    with fs.open(filename) as f:
        return pd.read_csv(f,delimiter=';',decimal='.')#.read().decode("utf-8")

col1, col2 = st.columns([0.8,0.2])
if groupe == 'phyling':
    if choose == 'CMJ':
            
        df=pd.DataFrame()
        for excel_file in fs.find("s3://phyling/"+choose+'/data'):
            if excel_file[-3:]=='csv':
                df=pd.concat([df,read_file(excel_file)],axis=0,ignore_index=True)
        df=df.dropna()
        
        if analyse == "Suivi d'indicateurs":

            with col1:
                st.header(analyse+' '+choose)
                st.subheader(groupe)

                # display the dataset
                if st.checkbox("Voir le tableau de données"):
                    st.write("### Enter the number of rows to view")
                    rows = st.number_input("", min_value=0,value=1)
                    if rows > 0:
                        st.dataframe(df.head(rows))

                sujet = df['athlete_name'].unique().tolist()
                sujet_list = st.multiselect("Athlète",sujet,default=sujet)
                mask = (df['athlete_name'].isin(sujet_list))
                selected_df = df[mask]

                selected_unique_sujet = selected_df['athlete_name'].unique().tolist()
                st.write('Nombre de sujet selectionnés : ',len(selected_unique_sujet))

                columns = selected_df.columns.tolist()
                selected_df.set_index(columns[3], inplace=True)

                columns = selected_df.columns.tolist()
                gp_df = selected_df.groupby('date').agg({'imp': 'mean',
                                                         'rel_imp': 'mean',
                                                         'imp_g': 'mean',
                                                         'rel_imp_g': 'mean',
                                                         'imp_d': 'mean',
                                                         'rel_imp_d': 'mean',
                                                         'P_moy': 'mean',
                                                         'rel_P_moy': 'mean',
                                                         'JH': 'mean',
                                                         'ass_d':'mean',
                                                         'ass_g':'mean',
                                                         'RSI': 'mean',
                                                         'CRPD': 'mean',
                                                         'ERFD': 'mean',
                                                         'Etime': 'mean',
                                                         'poids' : 'mean',
                                                         'Hauteur de saut':'mean',
                                                         'imp_mom_h':'mean',
                                                         'athlete_name': lambda x: x.iloc[0]})

                columns = ['imp','rel_imp','imp_g','rel_imp_g','imp_d',
                            'rel_imp_d','P_moy','rel_P_moy','JH','RSI','CRPD','ERFD','Etime',
                            'poids','Hauteur de saut','imp_mom_h']

                column_name = st.multiselect("Indicteurs",columns,default='Hauteur de saut')

                fig = px.line(gp_df, x=gp_df.index, y=column_name, markers=True,color='athlete_name')
                st.plotly_chart(fig,use_container_width=True)
                
                if st.checkbox("Analyse asymétrie"):
                    suj = st.selectbox("Athlete",sujet)
                    start = df[df['athlete_name']==suj]['date'].unique()[0]
                    end = df[df['athlete_name']==suj]['date'].unique()[-1]

                    start_t, end_t = st.select_slider("Selectionnez deux dates pour comparer l'évolution entre les deux",
                                                   options=df[df['athlete_name']==suj]['date'].unique(),
                                                   value=(start, end))
                    
                    labels=['droite','gauche']
                    colors=['#063f5b','#0C80BA']
                    gp_df=df[df['athlete_name']==suj].groupby('date').mean()
                    # Create subplots: use 'domain' type for Pie subplot
                    fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
                    fig.add_trace(go.Pie(labels=labels, values=[gp_df.loc[start_t]['ass_d'],gp_df.loc[start_t]['ass_g']]),
                                  1, 1)
                    fig.add_trace(go.Pie(labels=labels, values=[gp_df.loc[end_t]['ass_d'],gp_df.loc[end_t]['ass_g']]),
                                  1, 2)

                    # Use `hole` to create a donut-like pie chart
                    fig.update_traces(hole=.35, hoverinfo="label+percent+name",marker=dict(colors=colors))
                    fig.update_layout(margin=dict(t=25, b=0, l=0, r=0),
                        title_text="Asymétrie moyenne",
                        # Add annotations in the center of the donut pies.
                        annotations=[dict(text=start_t[0:10], x=0.16, y=0.5, font_size=12, showarrow=False),
                                     dict(text=end_t[0:10], x=0.84, y=0.5, font_size=12, showarrow=False)])
                    st.plotly_chart(fig,use_container_width=True)
                    
                
            with col2:
#                         st.image(logo, width=130 )
                        st.write("https://phyling.fr")
                
        elif analyse == 'Statistiques':
            with col1:
                st.header(analyse+' '+choose)
                st.subheader("Comparaison d'un athlète à la moyenne du groupe")
                sujet = df['athlete_name'].unique().tolist()
                sujet_list = st.selectbox("Athlete",sujet)
                columns = ['imp','rel_imp','imp_g','rel_imp_g','imp_d',
                            'rel_imp_d','P_moy','rel_P_moy','JH','RSI','CRPD','ERFD','Etime',
                            'poids','Hauteur de saut','imp_mom_h','imp_mom_h2']


                val = st.selectbox("Indicateur",columns,index=14)
            with col2:
#                 st.image(logo, width=130 )
                st.write("https://phyling.fr")
            col3, col4 = st.columns(2)
            res=stats.ttest_1samp(df[df['athlete_name']==sujet_list][val].values, df[val].mean())
            col3.metric('Moyenne du groupe',np.round(df[val].mean(),decimals=2), delta='', delta_color="normal")

            col4.metric('Moyenne '+sujet_list,np.round(df[df['athlete_name']==sujet_list][val].mean(),decimals=2),delta= np.round(df[df['athlete_name']==sujet_list][val].mean()-df[val].mean(),decimals=2), delta_color="normal")
            if res[1]<0.05:
                st.success('écart significatif à la moyenne du groupe')
            else:
                st.warning("Attention, l'écart à la moyenne n'est pas significatif")
                st.write('p_value = ',res[1].round(2))
            col1, col2 = st.columns([0.8,0.2])
            col3, col4,col5 = st.columns([0.4,0.4,0.2])
            with col1:
                st.write('')
                st.subheader("Analyse statistique de l'évolution")
                start = df[df['athlete_name']==sujet_list]['date'].unique()[0]
                end = df[df['athlete_name']==sujet_list]['date'].unique()[-1]
                if start == end : 
                    st.write("L'athlete n'a pas assez de données pour l'analyse statistique")
                else:
                            
                    start_t, end_t = st.select_slider("Selectionnez deux dates pour comparer l'évolution entre les deux",
                                                       options=df[df['athlete_name']==sujet_list]['date'].unique(),
                                                       value=(start, end))
                
                    df.set_index('date',inplace=True)
               
                    if len(df[df['athlete_name']==sujet_list].loc[end_t][val].values) != len(df[df['athlete_name']==sujet_list].loc[start_t][val].values):
                        st.write('échantillons de taille inégale')
                    else :
                        res1=stats.ttest_rel(df[df['athlete_name']==sujet_list].loc[end_t][val].values,
                                             df[df['athlete_name']==sujet_list].loc[start_t][val].values)

                        col3.metric(val + ' moyenne '+start_t,
                                    np.round(df[df['athlete_name']==sujet_list].loc[start_t][val].mean(),decimals=2),
                                    delta='', delta_color="normal")
                        col4.metric(val + ' moyenne '+end_t,
                                    np.round(df[df['athlete_name']==sujet_list].loc[end_t][val].mean(),decimals=2),
                                    delta=np.round(df[df['athlete_name']==sujet_list].loc[end_t][val].mean()-df[df['athlete_name']==sujet_list].loc[start_t][val].mean(),decimals=2), delta_color="normal")

#                         col1, col2 = st.columns([0.8,0.2])
                        with col1:
                            if res1[1]<0.05:
                                st.success('écart significatif à la moyenne du groupe')
                            else:
                                st.warning("Attention, l'écart à la moyenne n'est pas significatif")
                                st.write('p_value = ',res1[1].round(2))
                             
        elif analyse == 'Comparaisons de courbes':
            df_courbes=pd.DataFrame()
            for excel_file in fs.find("s3://phyling/"+choose+'/courbes/'):
                if excel_file[-3:]=='csv':
                    df_courbes=pd.concat([df_courbes,read_file_courbes(excel_file)],axis=0,ignore_index=True)
            df_courbes=df_courbes.dropna()
#             st.dataframe(df_courbes)
            with col1:
                st.header(analyse+' '+choose)
                st.subheader("Superposition de courbes de plusieurs athlètes")
                
                athlete1 = st.selectbox(
     'Athlète 1 :',
     (df_courbes['athlete_name'].unique()))
                
                date1=[]
                for d in df_courbes['date'][df_courbes['athlete_name']==athlete1].unique():
                    date1.append(d)
                if len(date1)>1:
                    d1 = st.select_slider(
         'Date athlète 1',
         options=date1)
                else :
                    d1=date1[0]
                
                athlete2 = st.selectbox(
     'Athlète 2 :',
     (df_courbes['athlete_name'].unique()))
                
                date2=[]
                for d in df_courbes['date'][df_courbes['athlete_name']==athlete2].unique():
                    date2.append(d)
                if len(date2)>1:
                    d2 = st.select_slider(
         'Date athlète 2',
         options=date2)
                else :
                    d2=date2[0]
                
#                 PLOT moyennes et std
                fig, ax = plt.subplots()
                athl_list=[athlete1,athlete2]
                d_list=[d1,d2]
            
                norm = st.checkbox('données normalisées')
                for i in range (0,2):
                    if norm:
                        c=df_courbes[df_courbes['athlete_name']==athl_list[i]]
                        c=c[c['date']==d_list[i]]
                        x=np.linspace(0,100,np.shape(c)[0])

                        ax.plot(x,c['moy'],label=athl_list[i])
                        ax.fill_between(x,c['moy'].values-c['std'],c['moy'].values+c['std'],alpha=0.15)
                        ax.set_xlabel("% Temps d'impulsion")
                        

                    else :
                        c=df_courbes[df_courbes['athlete_name']==athl_list[i]]
                        c=c[c['date']==d_list[i]]
                        x=np.linspace(0,np.shape(c)[0],np.shape(c)[0])

                        ax.plot(c['moy'].values,label=athl_list[i])
                        ax.fill_between(x,c['moy'].values-c['std'],c['moy'].values+c['std'],alpha=0.15)
                        ax.set_xlabel("Temps (ms)")
                        
                ax.grid(axis='y',alpha=0.3,linestyle='--')        
                ax.set_ylabel("Force Verticale (N)")
                ax.legend()

                st.pyplot(fig)

    if choose == 'Nordic Harmstring' :
        
        df=pd.DataFrame()
        for excel_file in fs.find("s3://phyling/"+choose):
            if excel_file[-3:]=='csv':
                df=pd.concat([df,read_file(excel_file)],axis=0,ignore_index=True)
        df=df.dropna()
        
        if analyse == "Suivi d'indicateurs":

            with col1:
                st.header(analyse+' '+choose)
                st.subheader(groupe)

                # display the dataset
                if st.checkbox("Voir le tableau de données"):
                    st.write("### Enter the number of rows to view")
                    rows = st.number_input("", min_value=0,value=1)
                    if rows > 0:
                        st.dataframe(df.head(rows))

                sujet = df['athlete_name'].unique().tolist()
                sujet_list = st.multiselect("Athlète",sujet,default=sujet)
                mask = (df['athlete_name'].isin(sujet_list))
                selected_df = df[mask]

                selected_unique_sujet = selected_df['athlete_name'].unique().tolist()
                st.write('Nombre de sujet selectionnés : ',len(selected_unique_sujet))

                columns = selected_df.columns.tolist()
                selected_df.set_index(columns[3], inplace=True)

                columns = selected_df.columns.tolist()
                gp_df = selected_df.groupby('date').agg({'F_max': 'mean',
                                                         'F_droite': 'mean',
                                                         'F_gauche': 'mean',
                                                         'athlete_name': lambda x: x.iloc[0]})

                columns = ['F_max','F_droite','F_gauche']

                column_name = st.multiselect("Indicteurs",columns,default='F_max')

                fig = px.line(gp_df, x=gp_df.index, y=column_name, markers=True,color='athlete_name')
                st.plotly_chart(fig,use_container_width=True)
                
            with col2:
#                         st.image(logo, width=130 )
                        st.write("https://phyling.fr")
                        

        
                        

        

                        

        
