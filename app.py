import dash
from dash import no_update
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import io
import requests
from mlserver.codecs import StringCodec

import base64

app = dash.Dash(
    __name__,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
app.title = "Semi Supervised Image classification"

server = app.server
app.config.suppress_callback_exceptions = True

model_list = [
    "LeNet5",
    "ResNet18",
]


def description_card():
    """

    :return: A Div containing dashboard title & descriptions.
    """
    return html.Div(
        id="description-card",
        children=[
            html.H5("Semi supervised Image classification "),
            html.H3("Welcome to Ssima model  "),
            html.Div(
                id="intro",
                children="Explore the model by uploading image and get the prediction. Model behind the scene is the"
                         "deep convolutional Neural Network train to recognize image Hand-writing images",
            ),
        ],
    )


def generate_control_card():
    """

    :return: A Div containing controls for graphs.

    content:
        - upload button for image
        - model selection & hyperparms
    """

    return html.Div(id="control-card",
                    children=[
                        html.P("Upload image file"),
                        dcc.Upload(
                            id="upload-image",
                            children=[
                                'Drag and Drop or ',
                                html.A('Select a File')
                            ],
                            style={
                                # "color": "darkgray",
                                "width": "100%",
                                "height": "50px",
                                "lineHeight": "50px",
                                "borderWidth": "1px",
                                "borderStyle": "dashed",
                                "borderRadius": "5px",
                                "borderColor": "darkgray",
                                "textAlign": "center",
                                "padding": "2rem 0",
                                "margin-bottom": "2rem"
                            },
                            # Allow multiple files to be uploaded
                            multiple=True
                            ),
                        html.Br(),

                        html.P("Select Model"),
                        dcc.Dropdown(
                            id="model-dropdown",
                            options=[{"label": i, "value": i} for i in model_list],
                            value=model_list[0],
                        ),
                        html.Br(),
                        html.Br(),
                        html.Div(
                            id="predict-btn-outer",
                            children=html.Button(id="predict-button",
                                                 children="Predict",
                                                 n_clicks=0),
                        ),

                        # Image Prediction
                        html.Br(),
                    ])


# App Layout
app.layout = html.Div(
    id="app-container",
    children=[
        # Banner
        html.Div(
            id="banner",
            className="banner",
            children=[html.Img(src=app.get_asset_url("plotly_logo.png"))],
        ),
        # Left column
        html.Div(
            id="left-column",
            className="four columns",
            children=[description_card(), generate_control_card()]
                     + 
                     [
                         html.Div(
                             ["initial child"], id="output-clientside", style={"display": "none"}
                         )
                     ],
        ),
        # Right column
        html.Div(
            id="right-column",
            className="eight columns",
        ),
    ],
)


def parse_contents(contents, filename, prediction):
    return html.Div([
        html.H5(f'Actual {filename}'),
        html.Img(src=contents),
        html.H5(f'Predicted {prediction}')
    ], className="grid-item")  # Ajouter la classe CSS grid-item


def send(addr, image_encoded_list):
    
    inference_request = {
        "inputs": [
            StringCodec.encode_input(name='payload', payload=image_encoded_list, use_bytes=False).model_dump()
        ]
    }
    
    print(inference_request)
    
    if image_encoded_list is not None:
        r = requests.post(addr, json=inference_request)

    if r.status_code == 200:
        return r.json()
    else:
        return {'message': f'Erreur lors de la requête POST. Code d\'état : {r.status_code}"'}


@app.callback(
    Output('right-column', 'children'),
    Input('predict-button', 'n_clicks'), 
    State('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('model-dropdown', 'value'),
    State('right-column', 'children')  # Récupérer l'état actuel du conteneur
)
def update_output(n_clicks, list_of_contents, list_of_names, selected_model, current_children):
    # Si le bouton n'a pas été cliqué, on ne fait rien
    if n_clicks is None or n_clicks == 0:
        return no_update

    # Si aucune image n'est téléchargée, on ne fait rien
    if list_of_contents is None:
        print("Aucune image n'a été téléchargée")
        return no_update

    # Si aucun modèle n'a été sélectionné, on ne fait rien
    if selected_model is None:
        print("Aucun modèle n'a été sélectionné")
        return no_update

    # Encode les nouvelles images (suppression de la partie "data:image/...base64,")
    list_of_images = [c.split(',')[1] for c in list_of_contents]

    addr = "http://localhost:8080/v2/models/mnist-resnet/infer"

    # Initialisation de list_of_preds par défaut à None
    list_of_preds = None

    try:
        # Api call
        r = send(addr, list_of_images)
        # get the predictions from the response if it exists and is valid 
        list_of_preds = r['outputs'][0]['data'] if r and 'outputs' in r else None
    except Exception as e:
        print(f"Erreur lors de l'appel à l'API : {e}")
        return no_update

    if list_of_preds is None or not isinstance(list_of_preds, list):
        print("Erreur : la réponse de l'API ne contient pas de prédictions valides")
        return no_update

    new_children = [
        parse_contents(c, n, p) for c, n, p in
        zip(list_of_contents, list_of_names, list_of_preds)
    ]

    if current_children is None:
        current_children = []

    return current_children + new_children



# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
