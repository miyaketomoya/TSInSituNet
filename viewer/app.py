import dash
from dash import dcc, html, callback_context
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import numpy as np
from viewerUtil import load_xyz,tensor_to_base64
from dash.exceptions import PreventUpdate
from generate_image import setup_model,generate_img
import base64
import os

app = dash.Dash(__name__)

def generate_sphere_data():
    """球面のデータを生成する関数"""
    radius = 3
    theta = np.linspace(0, 2 * np.pi, 100)
    phi = np.linspace(0, np.pi, 100)
    x = radius*np.outer(np.cos(theta), np.sin(phi))
    y = radius*np.outer(np.sin(theta), np.sin(phi))
    z = radius*np.outer(np.ones(100), np.cos(phi))
    return x, y, z

def create_3d_scatter_figure(x, y, z, opacity=0.5, marker_size=5):
    x_point, y_point, z_point = load_xyz("data/params_test.csv")
    fig = go.Figure(data=[
        go.Surface(z=z, x=x, y=y, colorscale='Blues', opacity=opacity),
        go.Scatter3d(
            x=x_point,
            y=y_point,
            z=z_point,
            mode='markers',
            marker=dict(size=marker_size, color='red')
        ),
        # X軸に沿った線
        go.Scatter3d(
            x=[0, 0],
            y=[min(y_point), max(y_point)],
            z=[min(z_point), max(z_point)],
            mode='lines',
            line=dict(color='blue', width=2)
        ),
        # Y軸に沿った線
        go.Scatter3d(
            x=[min(x_point), max(x_point)],
            y=[0, 0],
            z=[min(z_point), max(z_point)],
            mode='lines',
            line=dict(color='red', width=2)
        ),
        # Z軸に沿った線
        go.Scatter3d(
            x=[min(x_point), max(x_point)],
            y=[min(y_point), max(y_point)],
            z=[0, 0],
            mode='lines',
            line=dict(color='green', width=2)
        )
    ])
    fig.update_layout(
        title='3D Scatter Plot with Axis Lines',
        margin=dict(l=0, r=0, b=0, t=30),
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        )
    )
    return fig


#ここからスタート
logger_dir = "/home/tomoyam/TSInSituNet/logs/2024-05-30T16-37-53_TransformerSmokeRing_AddFirst"
model,device = setup_model(logger_dir)

x, y, z = generate_sphere_data()
initial_figure = create_3d_scatter_figure(x, y, z)

app.layout = html.Div([
    html.Div([
        html.Div([
            html.Label("viscosity:", style={'marginBottom': '20px'}),
            dcc.Slider(
                id='viscosity-slider',
                min=0.02,
                max=0.035,
                step=0.0005,
                value=0.025,
                marks={i/2000: f"{i/2000:.4f}" for i in range(40, 71)}, 
            ),
            html.Label("thermal_diffusivity:", style={'marginBottom': '20px'}),
            dcc.Slider(
                id='thermal_diffusivity-slider',
                min=0.03,
                max=0.03,
                step=None,
                value=0.03,
            ),
            html.Label("Select Timestep:", style={'marginBottom': '20px'}),
            dcc.Input(
                id='timestep-input',
                type='number',
                value=0,
                min=0,
                step=1,
                style={'marginBottom': '20px'}
            ),
            html.Div(id='coordinates-display', style={'marginTop': '20px', 'marginBottom': '20px'}),  # 座標を表示するDiv
            html.Button('Generate', id='generate-button', n_clicks=0, style={'marginBottom': '20px'}),
        ], style={'flex': '1'}),
        dcc.Graph(
            id='3d-plot',
            figure=initial_figure,
            config={'staticPlot': False},
            style={'height': '50%', 'width': '50%'}
        )
    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Div(id='output-data', style={'height': '50%'}),
    dcc.Store(id='selected-x'),
    dcc.Store(id='selected-y'),
    dcc.Store(id='selected-z')
])

@app.callback(
    [Output('3d-plot', 'figure'), Output('coordinates-display', 'children'),
     Output('selected-x', 'data'), Output('selected-y', 'data'), Output('selected-z', 'data')],
    [Input('3d-plot', 'clickData'), Input('generate-button', 'n_clicks')],
    [State('3d-plot', 'figure'), State('viscosity-slider', 'value'), State('thermal_diffusivity-slider', 'value'), State('timestep-input', 'value'),
     State('selected-x', 'data'), State('selected-y', 'data'), State('selected-z', 'data')]
)
def update_output(clickData, n_clicks, fig, viscosity, thermal_diffusivity, timestep, selected_x, selected_y, selected_z):
    triggered_id = callback_context.triggered[0]['prop_id'].split('.')[0]
    
    if triggered_id == '3d-plot':
        if clickData is None:
            coordinates_text = "No coordinates selected"
            return fig, coordinates_text, dash.no_update, dash.no_update, dash.no_update

        point = clickData['points'][0]
        curve_number = point['curveNumber']
        
        if curve_number != 1:
            coordinates_text = "No coordinates selected"
            return fig, coordinates_text, dash.no_update, dash.no_update, dash.no_update

        x, y, z = point['x'], point['y'], point['z']

        # 選択された点を黄色で表示
        fig['data'][1]['x'] = [x]
        fig['data'][1]['y'] = [y]
        fig['data'][1]['z'] = [z]
        fig['data'][1]['marker']['color'] = 'yellow'
        fig['data'][1]['marker']['size'] = 10

        coordinates_text = f"Selected point coordinates: x={x}, y={y}, z={z}"
        return fig, coordinates_text, x, y, z

    elif triggered_id == 'generate-button':
        if n_clicks > 0:
            img_list, code_block_list, timestep_list = generate_img(model, device, viscosity, thermal_diffusivity, timestep, selected_x, selected_y, selected_z, 50)
            
            # # 画像を自動的に保存
            for img, ts in zip(img_list, timestep_list):
                # ディレクトリパスを生成
                dir_path = f"output/v{viscosity}_t{thermal_diffusivity}_x{selected_x}_y{selected_y}_z{selected_z}"
                # ディレクトリが存在しない場合は作成
                os.makedirs(dir_path, exist_ok=True)
                # ファイル名をタイムステップのみに設定
                filename = f"{dir_path}/{ts}.png"
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(tensor_to_base64(img)))
            
            
            images_display = html.Div([
                html.Div([
                    html.Img(src='data:image/png;base64,' + tensor_to_base64(img), style={'width': '128px', 'height': '128px'}),
                    html.Div(f'Timestep: {timestep}', style={'textAlign': 'center'}),
                    # # 画像を保存するためのボタンを追加
                    # html.Button('Save Image', id=f'save-image-{timestep}', n_clicks=0)
                ]) for img, timestep in zip(img_list, timestep_list)
            ], style={'display': 'flex', 'flexDirection': 'row', 'overflowX': 'scroll'})
            
            # # 全ての画像を一気に保存するボタンを追加
            # images_display.children.append(
            #     html.Button('Save All Images', id='save-all-images', n_clicks=0)
            # )
            
            # # 画像を保存するためのコールバックを追加
            # for img, timestep in zip(img_list, timestep_list):
            #     @app.callback(
            #         Output(f'save-image-{timestep}', 'children'),
            #         [Input(f'save-image-{timestep}', 'n_clicks')],
            #         prevent_initial_call=True
            #     )
            #     def save_image(n_clicks):
            #         if n_clicks > 0:
            #             filename = f"output_{viscosity}_{thermal_diffusivity}_{timestep}.png"
            #             with open(filename, "wb") as f:
            #                 f.write(base64.b64decode(tensor_to_base64(img)))
            #             return f"Saved as {filename}"
            
            # # 全ての画像を保存するコールバック
            # @app.callback(
            #     Output('save-all-images', 'children'),
            #     [Input('save-all-images', 'n_clicks')],
            #     prevent_initial_call=True
            # )
            # def save_all_images(n_clicks):
            #     if n_clicks > 0:
            #         for img, timestep in zip(img_list, timestep_list):
            #             filename = f"viewer/output/output_{viscosity}_{thermal_diffusivity}_{timestep}.png"
            #             with open(filename, "wb") as f:
            #                 f.write(base64.b64decode(tensor_to_base64(img)))
            #         return "All images saved"
            
            return fig, images_display, dash.no_update, dash.no_update, dash.no_update
        else:
            return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

if __name__ == '__main__':
    app.run_server(debug=True)