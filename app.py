import streamlit as st
st.set_page_config(layout="wide")
import faiss
from faiss import METRIC_INNER_PRODUCT
import pandas as pd
import joblib
import numpy as np
import torch
from bipartite_models import TransRBipartiteModel
import requests
import plotly.graph_objects as go
import s3fs

class Args:
    datapath = 's3://qx-poc-public/recommender'
    modelpath = 's3://qx-poc-public/recommender/transrBipartite-marginloss0_5-800epoch-5neg'
    
FS = s3fs.S3FileSystem(anon=False)

def load(path):
    if 'df' in st.session_state and 'emb' in st.session_state and 'index' in st.session_state:
        df = st.session_state['df']
        emb = st.session_state['emb']
        index = st.session_state['index']
    else: 
        if FS is None:
            df = joblib.load(path+'/df.joblib')
            df = df.reset_index(drop=True).reset_index()
            df = df.sort_values(by='published_date', ascending=False)
            emb = np.load(path+'/embeds.npy')
        else:
            with FS.open(path+'/df.joblib') as f:
                df = joblib.load(f)
            with FS.open(path+'/embeds.npy') as f:
                emb = np.load(f)
        st.session_state['df'] = df
        st.session_state['emb'] = emb

        string_factory = 'IVF512,Flat'
        print('Building index...', end='')
        index = faiss.index_factory(384, string_factory, METRIC_INNER_PRODUCT)
    
    if not index.is_trained:
        index.train(emb)
        index.add(emb)
        index.nprobe = 12
        st.session_state['index'] = index

    return df, emb, index


def search(domain, rep_vectors, faiss_index, df, head2ix, embeddings, model, display_top_n=5, 
    search_n_per_signpost=5000, language='any', debug=False, favor='na'):
    favor = favor.split(',')
    if all([sn.isnumeric() for sn in favor]):
        favor = [int(sn) for sn in favor]
        _, scores, indices = faiss_index.range_search(embeddings[favor,:], 0.48)
    else:
        scores, indices = faiss_index.search(torch.vstack(rep_vectors['rep_vectors'][domain]).numpy(), 
            search_n_per_signpost)
    indices = list(set(indices.reshape(-1).tolist()))

    with torch.no_grad():
        h = head2ix[domain]
        te = torch.tensor(embeddings[indices], device='cpu')
        scores = model.scoring_function(
                h_idx=torch.tensor([h], device = 'cpu'),
                r_idx=torch.tensor([0], device = 'cpu'),
                t_idx=None,
                new_tails=te)
        scores = torch.tanh(scores+2.5)
        topn = torch.argsort(scores, descending=True)[:max(300, int(search_n_per_signpost/4))].tolist()

    indices_ = np.asarray(indices)[topn].tolist()
    scores_ = scores[topn].numpy().tolist()
    resultdf = df.loc[indices_].drop(columns=['media_item_id','name'])
    resultdf['score'] = scores_
    resultdf = resultdf.drop_duplicates(subset='title')
    if language != 'any':
        resultdf = resultdf[resultdf.language==language]
    resultdf = resultdf.drop(columns=['language'])
    try:
        resultdf = resultdf.head(display_top_n)
        resultdf['title'] = resultdf.title.apply(lambda x: x[:120])
        resultdf['content'] = resultdf.content.apply(lambda x: x[:500]+'...')
        return resultdf
    except Exception as e:
        print('topn ', topn[:10])
        print('indices ', indices[:10])
        if debug:
            raise(e)
        else:
            print(e)
            return topn, indices
    return

def render(container, container2, **kwargs):
    resultdf = search(**kwargs)

    if resultdf is None:
        raise "search failed"
    else:
        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[50,270,450,100,50],
                    header=dict(values=['Type','Title','Content','Date','Score'],
                        fill_color='lightsteelblue',
                        font_color='black',
                        font_size=15,
                        align='left'),
                    cells=dict(values=[resultdf.type, resultdf.title, resultdf.content, resultdf.published_date.dt.strftime('%Y-%m-%d %H:%M'), resultdf.score.round(3)],
                        fill_color='#EEEEEE',
                        font_size=13,
                        align='left')
                )
            ])
        fig.update_layout(
            margin=dict(l=20, r=20, t=5, b=5),height=250)
        container.plotly_chart(fig, use_container_width=True)

        ddf = kwargs['df']
        ddf = ddf[ddf.name==kwargs['domain']]
        if kwargs['language']!='any':
            ddf = ddf[ddf.language==kwargs['language']]
        ddf = ddf.head(50)
        fig = go.Figure(
            data=[
                go.Table(
                    columnwidth=[50,50,220,450,150],
                    header=dict(values=['Sn','Type','Title','Content','Date'],
                        fill_color='lightsteelblue',
                        font_color='black',
                        font_size=15,
                        align='left'),
                    cells=dict(values=[ddf['index'], ddf.type, ddf.title, ddf.content.apply(lambda x: x[:500]+'...'), ddf.published_date.dt.strftime('%Y-%m-%d %H:%M')],
                        fill_color='#EEEEEE',
                        font_size=13,
                        align='left')
                )
            ])
        fig.update_layout(
            margin=dict(l=20, r=20, t=5, b=5),height=550)
        container2.plotly_chart(fig, use_container_width=True)


def main(args):
    df, embeddings, index = load(args.datapath)
    languages = ['any','en','es','pt'] + sorted(list(df.dropna(subset=['language']).language.unique()))

    model, head2ix = TransRBipartiteModel.load_pretrained(args.modelpath, fh=FS)

    if FS is None:
        with open(args.modelpath+'/rep_vectors.pt', 'rb') as f:
            rep_vectors = torch.load(f)
    else:
        with FS.open(args.modelpath+'/rep_vectors.pt') as f:
            rep_vectors = torch.load(f)

    du = st.sidebar.selectbox(label = 'Select your domain unit', options=sorted(list(head2ix.keys())), 
        index=0, key=None, help=None)

    lang = st.sidebar.selectbox(label = 'Select your preferred language', options=languages)

    sn = st.sidebar.text_input(label = 'Enter the serial numbers of preferred news', value='E.g. 1254,5561')
    c1 = st.container()
    c1.subheader('Recommended Articles')
    c2 = st.container()
    c2.subheader('Daily Articles [As of 24 Nov 21]')
    render(container = c1, container2=c2, **{'domain':du, 'rep_vectors':rep_vectors, 'faiss_index':index, 'df':df, 
        'head2ix':head2ix, 'embeddings':embeddings, 'model':model, 'language':lang, 'favor':sn})


if __name__ == '__main__':
    args = Args()
    main(args)
