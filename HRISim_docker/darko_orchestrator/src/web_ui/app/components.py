import dash_bootstrap_components as dbc
from dash import html, dcc
from app import ids, styles

def get_title():

    return html.Div(
        [
            html.Img(src='assets/darko-logo.png', style={'width': '30%'}),
            html.Div(
                html.H1("Orchestrator"),
                style={'display': 'inline-block', 'margin-left': '10px', "text-align": "center", "vertical-align": "middle"}
            )
        ],
        style={"text-align": "center", "vertical-align": "middle"}
    )


def get_output_alert(text):
    return dbc.Alert(
            text,
            id=ids.output_alert_id,
            dismissable=True,
            is_open=False,
            fade=True,
            color="info"
    )

def get_current_action_table():
    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Current action")
                ])
            ),
            html.Tbody(
                html.Tr([
                    html.Td(id=ids.current_action_id)
                ])
            )
        ],
        style={"text-align": "center"}
    )

def get_status_table():
    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Robot tray"),
                    html.Th("Elapsed time"),
                ])
            ),
            html.Tbody(
                html.Tr([
                    html.Td(id=ids.robot_tray_id),
                    html.Td(id=ids.elapsed_time_id)
                ])
            )
        ],
        style={"text-align": "center"}
    )


def get_risk_table():
    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Risk"),
                    html.Th("Predicted"),
                    html.Th("Current"),
                ])
            ),
            html.Tbody(
                [
                    html.Tr([
                        html.Th("Navigation"),
                        html.Td(
                            html.Div(
                                html.Div(id=ids.predicted_navigation_risk_cell_id, style=styles.led_style),
                                style={'height': '100%', 'display': 'flex', 'align-items' : 'center', 'justify-content': 'center'}
                            )
                        ),
                        html.Td(
                            html.Div(
                                html.Div(id=ids.current_navigation_risk_cell_id, style=styles.led_style),
                                style={'height': '100%', 'display': 'flex', 'align-items' : 'center', 'justify-content': 'center'}
                            )
                        )
                    ]),
                    html.Tr([
                        html.Th("Manipulation"),
                        html.Td(
                            html.Div(
                                html.Div(id=ids.predicted_manipulation_risk_cell_id, style=styles.led_style),
                                style={'height': '100%', 'display': 'flex', 'align-items' : 'center', 'justify-content': 'center'}
                            )
                        ),
                        html.Td(
                            html.Div(
                                html.Div(id=ids.current_manipulation_risk_cell_id, style=styles.led_style),
                                style={'height': '100%', 'display': 'flex', 'align-items' : 'center', 'justify-content': 'center'}
                            )
                        )
                    ]),
                ]
            )
        ],
        style={"text-align": "center"}
    )

def get_reschedule_table():
    return dbc.Table(
        [
            html.Thead(
                html.Tr([
                    html.Th("Reschedule triggered")
                ])
            ),
            html.Tbody(
                html.Tr([
                    html.Td(
                        html.Div(
                            html.Div(id=ids.reschedule_trigger_id, style=styles.led_style),
                            style={'height': '100%', 'display': 'flex', 'align-items' : 'center', 'justify-content': 'center'}
                        )
                    )
                ])
            )
        ],
        style={"text-align": "center"}
    )

    


def get_input_mission_table(objects, trays):
    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th('')] + [html.Th(obj) for obj in objects]
                )
            )
        ] +
        [
            html.Tbody(
                [html.Tr(
                    [html.Th(tray)] + [html.Td(dbc.Input(type='number', min=0, max=10, value=0, id=f"{tray}-{obj}-spinner")) for obj in objects]) for tray in trays
                ]
            )
        ],
        bordered=True,
        style={"text-align": "center", "vertical-align": "middle"}
    )

def get_state_table(objects, trays):
    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [html.Th('')] + [html.Th(obj) for obj in objects]
                )
            )
        ] +
        [
            html.Tbody(
                [html.Tr(
                    [html.Th(tray)] + [html.Td("", id=f"{tray}-{obj}-state-cell") for obj in objects]) for tray in trays
                ]
            )
        ],
        style={"text-align": "center"}
    )

def get_publish_button():
    return dbc.Button('Publish', id=ids.publish_button_id, n_clicks=0, color="success")

def get_timer(id, interval, max_interval):
    return dcc.Interval(id=id, interval=interval, n_intervals=0, max_intervals=max_interval)


