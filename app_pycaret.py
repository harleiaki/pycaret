
# Imports
import pandas            as pd
import streamlit         as st

from io                     import BytesIO
from pycaret.classification import load_model, predict_model


@st.cache
def convert_df(df):
    return df.to_csv(index=False).encode('utf-8')

# Fun��o para converter o df para excel
@st.cache
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data


# Fun��o principal da aplica��o
def main():
    # Configura��o inicial da p�gina da aplica��o
    st.set_page_config(page_title = 'PyCaret', \
        layout="wide",
        initial_sidebar_state='expanded'
    )

    # T�tulo principal da aplica��o
    st.write("""## Escorando o modelo gerado no pycaret """)
    st.markdown("---")
    
    # Bot�o para carregar arquivo na aplica��o
    st.sidebar.write("## Suba o arquivo")
    data_file_1 = st.sidebar.file_uploader("Bank Credit Dataset", type = ['csv','ftr'])

    # Verifica se h� conte�do carregado na aplica��o
    if (data_file_1 is not None):
        df_credit = pd.read_feather(data_file_1)
        df_credit = df_credit.sample(50000)

        model_saved = load_model('LR Model Aula 5 062022')
        predict = predict_model(model_saved, data=df_credit)

        df_xlsx = to_excel(predict)
        st.download_button(label='=� Download',
                            data=df_xlsx ,
                            file_name= 'predict.xlsx')


if __name__ == '__main__':
	main()
