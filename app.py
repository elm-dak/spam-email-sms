import dash
from dash import dcc, html
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer
from dash_html_components import Img

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = dash.Dash(__name__)

app.layout = html.Div(style={'backgroundColor': '#808080', 'height': '100vh', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center'}, children=[
    html.Div(children=[
        Img(src="https://img.icons8.com/ios-filled/50/000000/sms.png", style={'display': 'inline-block'}),
        html.H1("Email/SMS Verification Spam", style={"textAlign": "center", "color": "black", "display": "inline-block", "marginLeft": "10px"})
    ]),
    dcc.Textarea(id='my-input', value='', style={'width': '600px', 'height': "100px", 'border-radius': '8px' , 'background-color': '#fee6e3' , 'border': '2px solid #111'}),
    html.Button('Predict', id='submit-val', n_clicks=0, style={"marginTop": "20px" , 'align-items': 'center', 'background-color': '#fee6e3', 'border': '2px solid #111', 'border-radius': '8px', 'box-sizing': 'border-box', 'color': '#111', 'cursor': 'pointer', 'display': 'flex', 'font-family': 'Inter,sans-serif', 'font-size': '16px', 'height': '48px', 'justify-content': 'center', 'line-height': '24px', 'max-width': '100%', 'padding': '0 25px', 'position': 'relative', 'text-align': 'center', 'text-decoration': 'none', 'user-select': 'none', '-webkit-user-select': 'none', 'touch-action': 'manipulation'}),
    html.H2(id='result', style={"color": "white" }),
    html.Div("© 2023 Dakouky ElMestapha.", style={"color": "black", "marginTop": "140px"})
    ])

@app.callback(
    dash.dependencies.Output('result', 'children'),
    [dash.dependencies.Input('submit-val', 'n_clicks')],
    [dash.dependencies.State('my-input', 'value')]
)
def update_output(n_clicks, value):
    if n_clicks > 0:
        transformed_sms = transform_text(value)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]
        if result == 1:
            return html.Span("Spam", style={'color': 'red'})
        else:
            return html.Span("Not Spam", style={'color': 'green'})

if __name__ == '__main__':
    app.run_server(debug=True)
