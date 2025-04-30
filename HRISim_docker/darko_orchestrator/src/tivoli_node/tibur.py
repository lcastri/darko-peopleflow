#!/usr/bin/env python3
 
import rospy
import numpy as np
import time
from hrisim_prediction_srvs.srv import GetRiskMap, GetRiskMapRequest, GetRiskMapResponse
from std_srvs.srv import Trigger, TriggerResponse
 

class TiburNodeService:

    def __init__(self):

        # Initialize the ROS node
        rospy.init_node('tibur_node_service')
        
        # Parameters
        self.update_frequency = 0.1  # Hz (ogni 10 secondi)
        
        # Data containers
        self.prediction_risk_matrix = None
        self.last_matrix_timestamp = None  # Timestamp dell'ultima lettura della matrice
        self.prediction_risk_matrix_names = []
        
        # Time steps in seconds for the columns in the risk matrix
        self.time_steps = [0, 40, 80, 120]  # seconds
        
        # Create service to handle risk value requests
        self.risk_service = rospy.Service('/get_interpolated_risk', Trigger, self.handle_risk_request)
        
        # Create service to provide the interpolated matrix
        self.matrix_service = rospy.Service('/get_interpolated_matrix', GetRiskMap, self.provide_interpolated_matrix)
        
        # Log initialization
        rospy.loginfo("TiburNodeService inizializzato. Iniziando a raccogliere dati ogni 10 secondi...")
        
        # Start periodic data collection
        self.timer = rospy.Timer(rospy.Duration(1.0/self.update_frequency), self.collect_risk_map)
 

    def collect_risk_map(self, event=None):

        """Periodically collect risk map data and store the timestamp"""
        tic = time.perf_counter()
        
        try:
            # Wait for service with timeout
            rospy.wait_for_service('/get_risk_map')
            # rospy.loginfo(f"service wait {time.perf_counter()-tic}")
                
            # Create service proxy
            get_risk_map = rospy.ServiceProxy('/get_risk_map', GetRiskMap)
            
            # Create empty request
            req = GetRiskMapRequest()
            
            # Call service
            tic = time.perf_counter()
            resp = get_risk_map(req)
            # rospy.loginfo(f"service call {time.perf_counter()-tic}")
            
            # Process response
            self.prediction_risk_matrix_names = list(resp.waypoint_ids)
            
            n_row = resp.n_waypoint
            n_col = resp.n_steps
            
            # Reshape risk matrix
            self.prediction_risk_matrix = np.array(resp.PDs, dtype=np.float64).reshape((n_row, n_col))
            
            # Store the current timestamp
            self.last_matrix_timestamp = rospy.Time.now()
            
            # Print dimensions and timestamp for reference
            # rospy.loginfo(f"Matrix dimensions: {n_row} rows (waypoints) x {n_col} columns (time steps)")
            # rospy.loginfo(f"Matrix timestamp: {self.last_matrix_timestamp.to_sec()}")
            rospy.loginfo(f"Matrix max value: {np.max(self.prediction_risk_matrix)}")
            
        except (rospy.ROSException, rospy.ServiceException) as e:
            rospy.logerr(f"Failed to collect risk map: {e}")
    

    def get_time_adjusted_interpolation(self, elapsed_seconds):
        """
        Calcola una matrice interpolata basata sul tempo trascorso dall'ultima lettura
        
        Args:
            elapsed_seconds (float): Secondi trascorsi dall'ultima lettura della matrice
            
        Returns:
            np.ndarray: Matrice interpolata aggiustata in base al tempo
        """
        if self.prediction_risk_matrix is None:
            rospy.logwarn("Nessuna matrice di rischio disponibile per l'interpolazione temporale")
            return None
        
        n_rows, n_cols = self.prediction_risk_matrix.shape
        
        # Verifica che ci siano sufficienti colonne
        if n_cols < 4:
            rospy.logerr(f"La matrice di rischio ha solo {n_cols} colonne, ne servono almeno 4 per l'interpolazione")
            return None
        
        # Inizializza la nuova matrice con le stesse dimensioni
        interpolated_matrix = np.zeros_like(self.prediction_risk_matrix)
        
        # Calcola il fattore di interpolazione basato sul tempo trascorso
        # Più tempo è passato, più ci avviciniamo ai valori futuri
        # Assumiamo che il tempo massimo di validità sia di 10 secondi (l'intervallo di aggiornamento)
        alpha = min(elapsed_seconds / 40.0, 1.0)
        # rospy.loginfo(f"Fattore di interpolazione temporale alpha: {alpha:.4f} (basato su {elapsed_seconds:.2f} secondi trascorsi)")
        
        # Prima colonna: interpolazione tra prima e seconda della matrice originale
        interpolated_matrix[:, 0] = self.prediction_risk_matrix[:, 0] + alpha * (self.prediction_risk_matrix[:, 1] - self.prediction_risk_matrix[:, 0])
        
        # Seconda colonna: interpolazione tra seconda e terza della matrice originale
        interpolated_matrix[:, 1] = self.prediction_risk_matrix[:, 1] + alpha * (self.prediction_risk_matrix[:, 2] - self.prediction_risk_matrix[:, 1])
        
        # Terza colonna: interpolazione tra terza e quarta della matrice originale
        interpolated_matrix[:, 2] = self.prediction_risk_matrix[:, 2] + alpha * (self.prediction_risk_matrix[:, 3] - self.prediction_risk_matrix[:, 2])
        
        # Quarta colonna: mantiene i valori della quarta colonna della matrice originale
        interpolated_matrix[:, 3] = self.prediction_risk_matrix[:, 3]
        
        # Stampa alcuni confronti esemplificativi
        # if n_rows > 0:
        #     sample_row = 0  # Prima riga come esempio
        #     original = self.prediction_risk_matrix[sample_row, :]
        #     interpolated = interpolated_matrix[sample_row, :]
        #     rospy.loginfo(f"Esempio di interpolazione temporale (waypoint {self.prediction_risk_matrix_names[sample_row]}):")
        #     rospy.loginfo(f"  Originale: {original}")
        #     rospy.loginfo(f"  Temporale: {interpolated}")
            
        return interpolated_matrix
    

    def provide_interpolated_matrix(self, req):
        """
        Servizio che fornisce la matrice interpolata basata sul tempo trascorso dall'ultima lettura
        """
        if self.prediction_risk_matrix is None or self.last_matrix_timestamp is None:
            rospy.logwarn("Matrice originale o timestamp non disponibili")
            return GetRiskMapResponse()
        
        # Calcola il tempo trascorso dall'ultima lettura della matrice
        current_time = rospy.Time.now()
        elapsed_seconds = (current_time - self.last_matrix_timestamp).to_sec()
        # rospy.loginfo(f"Richiesta matrice interpolata. Tempo trascorso dall'ultima lettura: {elapsed_seconds:.2f} secondi")
        
        # Calcola la matrice interpolata basata sul tempo
        interpolated_matrix = self.get_time_adjusted_interpolation(elapsed_seconds)
        
        if interpolated_matrix is None:
            rospy.logwarn("Impossibile calcolare la matrice interpolata")
            return GetRiskMapResponse()
        
        # Prepara la risposta
        response = GetRiskMapResponse()
        
        # Imposta i waypoint IDs
        response.waypoint_ids = self.prediction_risk_matrix_names
        
        # Imposta le dimensioni
        n_rows, n_cols = interpolated_matrix.shape
        response.n_waypoint = n_rows
        response.n_steps = n_cols
        
        # Appiattisci la matrice per la risposta
        response.PDs = interpolated_matrix.flatten().tolist()
        
        # rospy.loginfo(f"Fornita matrice interpolata temporale di dimensioni {n_rows}x{n_cols}")
        return response
    

    def interpolate_risk_value(self, waypoint_idx, time_seconds, elapsed_seconds=None):
        """
        Interpolate the risk value for a specific waypoint at the given time.
        Uses time-adjusted interpolation if elapsed_seconds is provided.
        
        Args:
            waypoint_idx (int): Index of the waypoint
            time_seconds (float): Time in seconds to interpolate for
            elapsed_seconds (float, optional): Seconds elapsed since last matrix read
            
        Returns:
            float: Interpolated risk value
        """
        # Se elapsed_seconds è fornito, usa l'interpolazione temporale
        if elapsed_seconds is not None:
            risk_matrix = self.get_time_adjusted_interpolation(elapsed_seconds)
        else:
            risk_matrix = self.prediction_risk_matrix
        
        if risk_matrix is None:
            rospy.logwarn("No risk matrix available for interpolation")
            return None
            
        # Handle time out of range
        if time_seconds < self.time_steps[0]:
            return risk_matrix[waypoint_idx, 0]
        elif time_seconds >= self.time_steps[-1]:
            return risk_matrix[waypoint_idx, -1]
            
        # Find the time steps to interpolate between
        for i in range(len(self.time_steps) - 1):
            if self.time_steps[i] <= time_seconds < self.time_steps[i + 1]:
                t0 = self.time_steps[i]
                t1 = self.time_steps[i + 1]
                v0 = risk_matrix[waypoint_idx, i]
                v1 = risk_matrix[waypoint_idx, i + 1]
                
                # Linear interpolation
                alpha = (time_seconds - t0) / (t1 - t0)
                interpolated_value = v0 + alpha * (v1 - v0)
                
                return interpolated_value
                
        # Should not reach here but just in case
        return None
    

    def handle_risk_request(self, req):
        """Handle risk value request service using the time-adjusted interpolation"""
        if self.prediction_risk_matrix is None or self.last_matrix_timestamp is None:
            return TriggerResponse(success=False, message="Risk matrix or timestamp not available")
        
        # Calcola il tempo trascorso dall'ultima lettura della matrice
        current_time = rospy.Time.now()
        elapsed_seconds = (current_time - self.last_matrix_timestamp).to_sec()
            
        # For demonstration, select a random waypoint and arrival time
        waypoint_idx = np.random.randint(0, len(self.prediction_risk_matrix_names))
        waypoint_name = self.prediction_risk_matrix_names[waypoint_idx]
        
        # Simulate arrival time between 0 and 120 seconds
        arrival_time = np.random.uniform(0, 120)
        
        # Get interpolated risk value with time adjustment
        risk_value = self.interpolate_risk_value(waypoint_idx, arrival_time, elapsed_seconds)
        
        if risk_value is None:
            return TriggerResponse(success=False, message="Failed to interpolate risk value")
            
        response_msg = (
            f"Waypoint: {waypoint_name}, "
            f"Arrival time: {arrival_time:.2f}s, "
            f"Time since last update: {elapsed_seconds:.2f}s, "
            f"Interpolated risk: {risk_value:.2f}"
        )
        # rospy.loginfo(response_msg)
        
        return TriggerResponse(success=True, message=response_msg)
    

try:
    node = TiburNodeService()
    rospy.loginfo("TiburNodeService è in esecuzione come servizio con interpolazione temporale. Premere Ctrl+C per terminare.")
    rospy.spin()
except Exception as e:
    rospy.logerr(f"Errore imprevisto in TiburNodeService: {e}")
    import traceback
    rospy.logerr(traceback.format_exc())