import rospy, dash, json, dash_bootstrap_components as dbc
from dash import html

orchestrator_started = rospy.get_param("/orchestrator_started")
while not orchestrator_started:
    rospy.loginfo("Waiting for the orchestrator...")
    rospy.sleep(1)
    orchestrator_started = rospy.get_param("/orchestrator_started")

path_to_static_data = "../static_data"
static_data_names = [
    "action_graph_nodes",
    "items"
]

static_data = {}
for static_data_name in static_data_names:
    with open(path_to_static_data + "/" + static_data_name + ".json", "r") as json_file:
        static_data[static_data_name] = json.load(json_file)

trays = static_data["items"]["trays"]
objects = static_data["items"]["objects"]

tray_object_list = [
    {"tray": trays[t], "object": objects[o]} for t in range(len(trays)) for o in range(len(objects)) 
]

action_nodes = list(static_data["action_graph_nodes"].keys())
action_graph_nodes_int  = {int(n[1:]):static_data['action_graph_nodes'][n] for n in action_nodes }

old_current_state = []
qfa = []
in_mission = False

app = dash.Dash("darko-web-ui", external_stylesheets=[dbc.themes.BOOTSTRAP], prevent_initial_callbacks='initial_duplicate')

app.title = 'Darko Orchestrator UI'
app._favicon = 'darko-fav.png'

from app import ros_pub_sub
from app import components, ids

app.layout = dbc.Container([
    html.Hr(),
    components.get_title(),
    html.Hr(),
    components.get_output_alert(""),
    components.get_status_table(),
    dbc.Row([
            dbc.Col(
                components.get_risk_table()
            ),
            dbc.Col(
                components.get_reschedule_table()    
            )
    ]),
    components.get_current_action_table(),
    components.get_state_table(objects, trays),
    html.Hr(),
    html.Div([
        components.get_input_mission_table(objects, trays),
        components.get_publish_button(),  
    ], id=ids.input_div_id),
    components.get_timer(ids.update_ui_timer_id, 1000, -1),
    components.get_timer(ids.hide_reschedule_trigger_timer_id, 7000, 0)
])

from app import callbacks
