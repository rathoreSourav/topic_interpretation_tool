import seaborn as sns
import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import streamlit as st

st.set_option('deprecation.showPyplotGlobalUse', False)

def word_count_dist_by_topic():
    st.header("Distribution of Documents by Topic Number ")
    cols = [color for name, color in mcolors.XKCD_COLORS.items()][:50]  # more colors: 'mcolors.XKCD_COLORS'

    fig, axes = plt.subplots(nrows=13, ncols=4, figsize=(16,7), dpi=160)
    #axes = axes.flatten()

    df_dominant_topic = pd.read_csv("./output/extracted_doc.csv")
    topic_count=len(set(df_dominant_topic['Topic_Num'].unique()))
    #st.write(cols) 
    for i, ax in enumerate(axes.flatten()):   
        if i < topic_count:
          
            df_dominant_topic_sub = df_dominant_topic.loc[df_dominant_topic.Topic_Num == i, :]
            doc_lens = [len(str(d)) for d in df_dominant_topic_sub.Text]
            ax.hist(doc_lens, bins = 1000, color=cols[i])
            ax.tick_params(axis='y', labelcolor=cols[i], color=cols[i])
            sns.kdeplot(doc_lens, color="black", shade=False, ax=ax.twinx())
            ax.set(xlim=(0, 1000), xlabel='Document Word Count')
            ax.set_ylabel('Number of Documents', color=cols[i])
            ax.set_title('Topic: '+str(i), fontdict=dict(size=16, color=cols[i]))
        else:
            ax.remove()
            
 
   # axes[-1].axis('off')    

    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    plt.xticks(np.linspace(0,10))
    fig.suptitle('Distribution of Document Word Counts by Dominant Topic', fontsize=22)
    # set figure size
    fig.set_figheight(60)
    fig.set_figwidth(30)
    st.pyplot(fig)


def word_count_dist_whole():
    st.header("Distribution of Documents")
    df_dominant_topic = pd.read_csv("./output/extracted_doc.csv")
    doc_lens = [len(str(d)) for d in df_dominant_topic['Text']]

# Plot
    plt.figure(figsize=(16,7), dpi=160)
    plt.hist(doc_lens, bins = 4000, color='navy')
    plt.text(3500, 100, "Mean   : " + str(round(np.mean(doc_lens))))
    plt.text(3500,  90, "Median : " + str(round(np.median(doc_lens))))
    plt.text(3500,  80, "Stdev   : " + str(round(np.std(doc_lens))))
    plt.text(3500,  70, "1%ile    : " + str(round(np.quantile(doc_lens, q=0.01))))
    plt.text(3500,  60, "99%ile  : " + str(round(np.quantile(doc_lens, q=0.99))))

    plt.gca().set(xlim=(0, 4000), ylabel='Number of Documents', xlabel='Document Word Count')
    plt.tick_params(size=16)
    plt.xticks(np.linspace(0,4000,8))
    plt.title('Distribution of Document Word Counts', fontdict=dict(size=22))
    st.pyplot(plt.show())
