import rospy
import pandas as pd
import numpy as np
from dash import Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from app import ids, styles, app, objects, trays, tray_object_list, action_graph_nodes_int, old_current_state, in_mission, qfa, t_list
from app.ros_pub_sub import mission_pub, current_action_sub, current_state_sub, qfa_sub, monitoring_risk_sub, reschedule_sub, heatmap_subscriber
from std_msgs.msg import Int64MultiArray

def publish_mission(spinner_value_list):
    msg = Int64MultiArray()
    msg.data = spinner_value_list
    mission_pub._publish_msg(msg)

@app.callback(
    Output(ids.output_alert_id, 'children', allow_duplicate=True),
    Output(ids.output_alert_id, 'is_open', allow_duplicate=True),
    Output(ids.output_alert_id, 'color', allow_duplicate=True),
    Output(ids.publish_button_id, 'n_clicks'),
    Input(ids.publish_button_id, 'n_clicks'),
    [
        State(f"{to['tray']}-{to['object']}-spinner", "value") for to in tray_object_list
    ]
)
def update_output(n_clicks, *spinner_value_list):

    global in_mission
    global qfa

    if n_clicks > 0:

        if any(value > 0 for value in spinner_value_list):

            publish_mission(spinner_value_list)
            
            qfa = []

            while qfa_sub._check_empty_data():
                rospy.sleep(0.2)

            qfa = list(qfa_sub._data.state)
            qfa_sub._reset_data()

            in_mission = True
            return "Mission published", True, "info", 0
        
        return "Mission is empty.", True, "warning", 0
        

    return '', False, "info", 0

@app.callback(
    Output(ids.input_div_id, 'hidden'),
    Input(ids.update_ui_timer_id, 'n_intervals')
)
def update_input_div(n_intervals):

    global in_mission

    if in_mission:
        return True
    
    return False


@app.callback(
    Output(ids.current_action_id, 'children'),
    Input(ids.update_ui_timer_id, 'n_intervals')
)
def update_current_action(n_intervals):

    data = current_action_sub._data

    first_action = []
    second_action = []
    if data is not None:
        first_action = data.first_action.action
        second_action = data.second_action.action

    current_action_str = process_current_action(first_action, second_action)

    return current_action_str


@app.callback(
    Output(ids.reschedule_trigger_id, 'style'),
    Output(ids.hide_reschedule_trigger_timer_id, 'max_intervals'),
    Output(ids.hide_reschedule_trigger_timer_id, 'n_intervals'),
    Input(ids.update_ui_timer_id, 'n_intervals'),
    Input(ids.hide_reschedule_trigger_timer_id, 'n_intervals'),
    State(ids.hide_reschedule_trigger_timer_id, 'max_intervals'),
)
def update_reschedule_trigger(update_count, hide_count, hide_is_active):

    red = {**styles.led_style, **{'backgroundColor': 'red'}}
    white = {**styles.led_style, **{'backgroundColor': 'white'}}

    label_on = (red, 1, 0)
    label_off = (white, 0, 0)

    if hide_is_active > 0: 

        if hide_count > 0:
            return label_off
        
        return label_on
    
    if not reschedule_sub._check_empty_data():
        reschedule_sub._reset_data()
        return label_on
    
    return label_off

@app.callback(
    Output(ids.predicted_navigation_risk_cell_id, 'style'),
    Output(ids.current_navigation_risk_cell_id, 'style'),
    Output(ids.predicted_manipulation_risk_cell_id, 'style'),
    Output(ids.current_manipulation_risk_cell_id, 'style'),
    Input(ids.update_ui_timer_id, 'n_intervals')
)
def update_monitoring_risk(n_intervals):

    data = monitoring_risk_sub._data

    if in_mission and data is not None and not data == []:

        r0, g0, b0 = get_rgb_by_value(data[0])
        r1, g1, b1 = get_rgb_by_value(data[1])
        r2, g2, b2 = get_rgb_by_value(data[2])
        r3, g3, b3 = get_rgb_by_value(data[3])

        rgb0 = {'backgroundColor': f'rgb({r0},{g0},{b0})'}
        rgb1 = {'backgroundColor': f'rgb({r1},{g1},{b1})'}
        rgb2 = {'backgroundColor': f'rgb({r2},{g2},{b2})'}
        rgb3 = {'backgroundColor': f'rgb({r3},{g3},{b3})'}

        retval0 = {**styles.led_style, **rgb0}
        retval1 = {**styles.led_style, **rgb1}
        retval2 = {**styles.led_style, **rgb2}
        retval3 = {**styles.led_style, **rgb3}

        return retval0, retval1, retval2, retval3

    white = {'backgroundColor': 'rgb(255, 255, 255)'}
    retval = {**styles.led_style, **white}

    return retval, retval, retval, retval


@app.callback(
    Output(ids.elapsed_time_id, 'children'),
    Output(ids.robot_tray_id, 'children'),
    Output(ids.output_alert_id, 'children', allow_duplicate=True),
    Output(ids.output_alert_id, 'is_open', allow_duplicate=True),
    Output(ids.output_alert_id, 'color', allow_duplicate=True),
    [
        Output(f"{to['tray']}-{to['object']}-state-cell", "children") for to in tray_object_list
    ],
    [
        Output(f"{to['tray']}-{to['object']}-state-cell", "style") for to in tray_object_list
    ],
    Input(ids.update_ui_timer_id, 'n_intervals'),
    State(ids.elapsed_time_id, 'children'),
    State(ids.output_alert_id, 'children'),
    State(ids.output_alert_id, 'is_open'),
    State(ids.output_alert_id, 'color')
)
def update_current_state(n_intervals, elapsed_time, alert_text, alert_is_open, alert_color):

    global old_current_state
    global in_mission

    white = {'backgroundColor': 'rgb(255, 255, 255)'}

    current_state = list(current_state_sub._data.state)

    time = 0
    robot = "empty"
    state_cell_value_list = [""] * len(tray_object_list)
    state_cell_color_list = [white] * len(tray_object_list)

    if in_mission:

        time = str(int(elapsed_time if elapsed_time else -1) + 1)
        state_cell_value_list, state_cell_color_list = get_values_and_color_from_state([0] * len(qfa))
        
        if current_state:

            if old_current_state == current_state:
                time = str(int(elapsed_time) + 1)
            else:
                time = current_state[0]
                old_current_state = current_state

            robot = get_robot_tray_from_state(current_state[2:])

            state_cell_value_list, state_cell_color_list = get_values_and_color_from_state(current_state[2:])

            if qfa == current_state[2:]:

                alert_text = f"Mission completed in {current_state[0]} seconds."
                alert_is_open = True
                alert_color = "success"
                in_mission = False

    else:

        if qfa:
            state_cell_value_list, state_cell_color_list = get_values_and_color_from_state(qfa)

    return [time, robot, alert_text, alert_is_open, alert_color] + state_cell_value_list + state_cell_color_list

@app.callback(
    Output(ids.costmap_heatmap_graph, 'figure'),
    Input(ids.update_heatmap_timer_id, 'n_intervals'),
    State(ids.costmap_heatmap_graph, 'figure')
)
def update_costmap_heatmap(n_intervals, old_figure):
    
    hm_data = heatmap_subscriber._data

    if not hm_data or len(hm_data.heatmap_list) < 4:
        return old_figure

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f"t = {t}" for t in t_list],
        horizontal_spacing=0.02,
        vertical_spacing=0.06
    )

    for i in range(4):
        hm = hm_data.heatmap_list[i]
        row = i // 2 + 1
        col = i % 2 + 1

        fig.add_trace(go.Heatmap(
            z=list(hm.z),
            x=list(hm.x),
            y=list(hm.y),
            colorscale='Viridis',
            zmin=0,
            zmax=100,
            showscale=False if i > 0 else True  # Mostra la scala solo per il primo
        ), row=row, col=col)

    fig.update_layout(
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )

    # Nasconde assi e griglia per tutti i subplot
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_xaxes(showgrid=False, visible=False, row=i, col=j)
            fig.update_yaxes(showgrid=False, visible=False, row=i, col=j)

    return fig


def process_current_action(first_action_str_list, second_action_str_list):

    if not first_action_str_list:
        return 'Waiting...'

    first_action_type = first_action_str_list[0]

    if first_action_type == "moving":

        node = int(first_action_str_list[1])
        node_coords = f"({action_graph_nodes_int[node]['x']:.2f}, {action_graph_nodes_int[node]['y']:.2f})"

        if second_action_str_list:

            second_action_type = second_action_str_list[0]

            if second_action_type == "placing":

                object_index = int(second_action_str_list[1])
                tray_index = int(second_action_str_list[2])

                return f"Moving in {node_coords} to throw object {objects[object_index]} in tray {trays[tray_index]}"
            
            elif second_action_type == "picking":

                object_index = int(second_action_str_list[1])

                return f"Moving in {node_coords} to pick object {objects[object_index]}"
            
        return f"Moving in {node_coords}"

    elif first_action_type == "placing":

        object_index = int(first_action_str_list[1])
        tray_index = int(first_action_str_list[2])

        return f"Throwing object {objects[object_index]} in tray {trays[tray_index]}"

    elif first_action_type == "picking":

        object_index = int(first_action_str_list[1])

        return f"Picking object {objects[object_index]}"

    return 'Waiting...'


def get_values_and_color_from_state(current_state):

    global qfa

    warning = {'backgroundColor': 'rgb(255, 243, 205)'}
    success = {'backgroundColor': 'rgb(209, 231, 221)'}
    white = {'backgroundColor': 'rgb(255, 255, 255)'}

    n_trays = len(trays)

    state_cell_value_list, state_cell_color_list = [], []

    for t in range(n_trays):
        for o in range(len(objects)):
            index = (n_trays + 1) * o + t + 1
            current = current_state[index]
            total = qfa[index]
            state_cell_value_list += [f"{current}/{total}"]

            color = white
            if total > 0:
                if current == total:
                    color = success
                else:
                    color = warning

            state_cell_color_list += [color]

    return state_cell_value_list, state_cell_color_list


def get_robot_tray_from_state(state):

    n_trays = len(trays)

    for o in range(len(objects)):

        placed = sum([state[(n_trays + 1) * o + t + 1] for t in range(n_trays)])
        picked = state[(n_trays + 1) * o]

        if picked > placed:
            return objects[o]
        
    return "empty"
        

def get_rgb_by_value(value):

    white = (255, 255, 255)

    if value < 0:
        value = 0

    if value > 100:
        value = 100
    
    green = (0, 255, 0)
    yellow = (255, 255, 0)
    red = (255, 0, 0)
    
    if value < 50:
        r = green[0] + int((yellow[0] - green[0]) * (value / 50))
        g = green[1] + int((yellow[1] - green[1]) * (value / 50))
        b = green[2] + int((yellow[2] - green[2]) * (value / 50))
    else:
        r = yellow[0] + int((red[0] - yellow[0]) * ((value - 50) / 50))
        g = yellow[1] + int((red[1] - yellow[1]) * ((value - 50) / 50))
        b = yellow[2] + int((red[2] - yellow[2]) * ((value - 50) / 50))

    return r, g, b

