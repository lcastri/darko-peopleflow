import os
import pandas as pd
import copy
from enum import Enum

class TOD(Enum):
    STARTING = "STARTING"
    POSTER = "POSTER"
    BUFFET = "BUFFET"
    OFF = "OFF"

TODS = {t.value: i for i, t in enumerate(TOD)}

class WP(Enum):
    ROOM1 = "r1"
    ROOM2 = "r2"
    CORRIDOR1 = "c1"
    CORRIDOR2 = "c2"
    CORRIDOR3 = "c3"
    CORRIDOR4 = "c4"
    CORRIDOR5 = "c5"
    CORRIDOR6 = "c6"
    CORRIDOR7 = "c7"
    CORRIDOR8 = "c8"
    CORRIDOR9 = "c9"
    CORRIDOR10 = "c10"
    CORRIDOR11 = "c11"
    CORRIDOR12 = "c12"
    CORRIDOR13 = "c13"
    CORRIDOR14 = "c14"
    CORRIDOR15 = "c15"
    CORRIDOR16 = "c16"
    CORRIDOR17 = "c17"
    CORRIDOR18 = "c18"
    CORRIDOR19 = "c19"
    CORRIDOR20 = "c20"
    CORRIDOR21 = "c21"
    CORRIDOR22 = "c22"
    CORRIDOR23 = "c23"
    CORRIDOR24 = "c24"
    CORRIDOR25 = "c25"
    CORRIDOR26 = "c26"

WPS = {wp.value: i for i, wp in enumerate(WP)}
ID_TO_WP = {v: k for k, v in WPS.items()}  
  

def get_initrow(df):
    for r in range(len(df)):
        if (df.iloc[r]["G_X"] != -1000 and df.iloc[r]["G_Y"] != -1000 and df.iloc[r]["R_B"] != 0 and df.iloc[r].notnull().all()):
            return r


SF = 10 #Hz
INDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/csv/'
OUTDIR = '/home/hrisim/ros_ws/src/HRISim/postprocessing/hrisim_postprocess/csv_pp-2/'
BAGNAME= 'DISCOVERY'
STEP = 1
        
for tod in TOD:
    if tod is TOD.STARTING: continue
    print(f"Postprocessing {BAGNAME}_{tod.value}")
    DF = pd.read_csv(os.path.join(INDIR, f"{BAGNAME}_{tod.value}", f"{BAGNAME}_{tod.value}.csv"))
    
    # Remove initial rows with missing or invalid data
    r = get_initrow(DF)
    DF = pd.DataFrame(DF.values[r:,:], columns=DF.columns)
    DF.reset_index(drop=True, inplace=True)
    
    ###########################################################################
    # Subsampling
    ###########################################################################
    df_sub = DF.iloc[::STEP].copy()
    df_sub.reset_index(drop=True, inplace=True)
    
    ###########################################################################
    # Postprocessing
    ###########################################################################
    df_sub['BC'] = -df_sub['R_B'].diff()    
    df_sub['BC'] = df_sub['BC'].fillna(0.0)
    
    df_sub['DIST'] = ((df_sub['R_X']-df_sub['G_X'])**2 + (df_sub['R_Y']-df_sub['G_Y'])**2)**0.5
    
    ###########################################################################
    # FILTERING STRATEGY: Remove acceleration and braking phases to focus on cruising behavior
    ###########################################################################
    # # Filter out rows where DIST <= 1.0 to avoid including data points where the robot is slowing down near the goal
    # THRESHOLD = 1.0
    # df_sub = df_sub[df_sub['DIST'] > THRESHOLD]
    # df_sub.reset_index(drop=True, inplace=True)
    ROWS_TO_SKIP_ON_START = 15  
    BRAKING_DIST_THRESHOLD = 1.0 
    
    # 1. Crea un ID univoco ogni volta che la DESTINAZIONE cambia
    # Controlla se G_X è diverso dal G_X precedente, o se G_Y è diverso.
    goal_changed = (df_sub['G_X'] != df_sub['G_X'].shift()) | (df_sub['G_Y'] != df_sub['G_Y'].shift())
    df_sub['goal_group'] = goal_changed.cumsum()
    
    # 2. Conta le righe dall'inizio di questo specifico tragitto verso il goal
    df_sub['goal_row_number'] = df_sub.groupby('goal_group').cumcount()
    
    # 3. Definisci le maschere booleane per le righe DA TENERE
    is_past_acceleration = df_sub['goal_row_number'] >= ROWS_TO_SKIP_ON_START
    is_not_braking = df_sub['DIST'] > BRAKING_DIST_THRESHOLD
    
    # 4. Applica il filtro: tiene solo la fase di "Crociere"
    df_sub = df_sub[is_past_acceleration & is_not_braking].copy()
    
    # 5. Pulizia colonne temporanee
    df_sub.drop(columns=['goal_group', 'goal_row_number'], inplace=True)
    df_sub.reset_index(drop=True, inplace=True)
    
    
    for wp in WP:
        WDDF = copy.deepcopy(df_sub)
        WDDF['PD_WP'] = df_sub[f"{wp.value}_PD"]
        WDDF['WP'] = WPS[wp.value]
        
        # Filter rows where R_WP == WP
        WDDF = WDDF[WDDF['R_WP'] == WPS[wp.value]]
            
        # Save specific columns      
        WDDF.rename(columns={'TOD': 'S', 'R_WP': 'R_W', 'WP': 'W', 'R_V': 'V', 'R_B': 'B', 'OBS': 'O', 'PD_WP': 'D', 'BC': 'L'}, inplace=True)
        selected_clmns = ['S', 'W', 'V', 'B', 'L', 'O', 'D', 'DIST']
           
        output_dir = os.path.join(OUTDIR, f"{BAGNAME}_{tod.value}")
        os.makedirs(output_dir, exist_ok=True)
        WDDF[selected_clmns].to_csv(os.path.join(output_dir, f"{BAGNAME}_{tod.value}_{wp.value}.csv"), index=False)
        print(f"Saved processed data for {BAGNAME}_{tod.value}_{wp.value}.csv")
