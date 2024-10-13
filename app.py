import dash
from dash import no_update
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import io
import requests

import numpy as np


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
                     + [
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


def post_files_list_to_api(image_encoded_list: list = None,
                           model_name: str = 'LeNet5'):

    url = "http://127.0.0.1:8001/prediction"  # Your FastAPI URL
    
    # Properly format the files as expected by the API
    files = [
        ('files', (f'image_{i}.png', io.BytesIO(image_encoded), 'image/png')) 
        for i, image_encoded in enumerate(image_encoded_list)
    ]
    
    data = {'model_id': model_name}  # Ensure the form field matches the back-end

    print(f'===> {image_encoded_list}')
    if image_encoded_list is not None:
        responses_from_api = requests.post(url, files=files, data=data)

    if responses_from_api.status_code == 200:
        return responses_from_api.json()
    else:
        return {'message': f'Erreur lors de la requête POST. Code d\'état : {responses_from_api.status_code}"'}

@app.callback(
    Output('right-column', 'children'),
    Input('predict-button', 'n_clicks'), 
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('model-dropdown', 'value'),
    State('right-column', 'children')  # Keep track of the current children
)
def update_output(n_clicks, list_of_contents, list_of_names, selected_model, current_children):
    # if no image uploaded or no button clicked
    if list_of_contents is None or n_clicks is None:
        return no_update

    # Encode the images to base64 strings for API call 
    list_of_images = [base64.b64decode(c.split(',')[1]) for c in list_of_contents]
    
    # Api call
    preds_list_form_api = post_files_list_to_api(list_of_images, selected_model)
    
    print(f'===> Response from API: {preds_list_form_api}')
    
    if 'predictions' not in preds_list_form_api:
        print("Erreur : la réponse de l'API ne contient pas de 'predictions'")
        return no_update

    new_children = [
        parse_contents(c, n, p) for c, n, p in
        zip(list_of_contents, list_of_names, list(preds_list_form_api['predictions']))
    ]

    if current_children is None:
        current_children = []

    return current_children + new_children



# Run the server
if __name__ == "__main__":
    app.run_server(debug=True)
