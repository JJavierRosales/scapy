# Declare dictionary containing the link between JSpOC format and official 
# CCSDS
kelvins_to_ccsds_features = {
    'relative_speed': 'RELATIVE_SPEED',
    'ccsds_cdm_vers': 'CCSDS_CDM_VERS',
    'creation_date': 'CREATION_DATE',
    'originator':'ORIGINATOR',
    'message_for':'MESSAGE_FOR',
    'message_id':'MESSAGE_ID',
    'tca':'TCA',
    'miss_distance':'MISS_DISTANCE',
    'relative_speed':'RELATIVE_SPEED',
    'relative_position_r':'RELATIVE_POSITION_R',
    'relative_position_t':'RELATIVE_POSITION_T',
    'relative_position_n':'RELATIVE_POSITION_N',
    'relative_velocity_r':'RELATIVE_VELOCITY_R',
    'relative_velocity_t':'RELATIVE_VELOCITY_T',
    'relative_velocity_n':'RELATIVE_VELOCITY_N',
    'start_screen_period':'START_SCREEN_PERIOD',
    'stop_screen_period':'STOP_SCREEN_PERIOD',
    'screen_volume_frame':'SCREEN_VOLUME_FRAME',
    'screen_volume_shape':'SCREEN_VOLUME_SHAPE',
    'screen_volume_x':'SCREEN_VOLUME_X',
    'screen_volume_y':'SCREEN_VOLUME_Y',
    'screen_volume_z':'SCREEN_VOLUME_Z',
    'screen_entry_time':'SCREEN_ENTRY_TIME',
    'screen_exit_time':'SCREEN_EXIT_TIME',
    'jspoc_probability':'COLLISION_PROBABILITY',
    't_object_designator':'OBJECT1_OBJECT_DESIGNATOR',
    't_catalog_name':'OBJECT1_CATALOG_NAME',
    't_object_name':'OBJECT1_OBJECT_NAME',
    't_international_designator':'OBJECT1_INTERNATIONAL_DESIGNATOR',
    't_object_type':'OBJECT1_OBJECT_TYPE',
    't_ephemeris_name':'OBJECT1_EPHEMERIS_NAME',
    't_covariance_method':'OBJECT1_COVARIANCE_METHOD',
    't_maneuverable':'OBJECT1_MANEUVERABLE',
    't_orbit_center':'OBJECT1_ORBIT_CENTER',
    't_ref_frame':'OBJECT1_REF_FRAME',
    't_gravity_model':'OBJECT1_GRAVITY_MODEL',
    't_atmospheric_model':'OBJECT1_ATMOSPHERIC_MODEL',
    't_n_body_perturbations':'OBJECT1_N_BODY_PERTURBATIONS',
    't_solar_rad_pressure':'OBJECT1_SOLAR_RAD_PRESSURE',
    't_earth_tides':'OBJECT1_EARTH_TIDES',
    't_intrack_thrust':'OBJECT1_INTRACK_THRUST',
    't_time_lastob_start':'OBJECT1_TIME_LASTOB_START',
    't_time_lastob_end':'OBJECT1_TIME_LASTOB_END',
    't_recommended_od_span':'OBJECT1_RECOMMENDED_OD_SPAN',
    't_actual_od_span':'OBJECT1_ACTUAL_OD_SPAN',
    't_obs_available':'OBJECT1_OBS_AVAILABLE',
    't_obs_used':'OBJECT1_OBS_USED',
    't_tracks_available':'OBJECT1_TRACKS_AVAILABLE',
    't_tracks_used':'OBJECT1_TRACKS_USED',
    't_residuals_accepted':'OBJECT1_RESIDUALS_ACCEPTED',
    't_weighted_rms':'OBJECT1_WEIGHTED_RMS',
    't_area_pc':'OBJECT1_AREA_PC',
    't_area_drg':'OBJECT1_AREA_DRG',
    't_area_srg':'OBJECT1_AREA_SRP',
    't_mass':'OBJECT1_MASS',
    't_cd_area_over_mass':'OBJECT1_CD_AREA_OVER_MASS',
    't_cr_area_over_mass':'OBJECT1_CR_AREA_OVER_MASS',
    't_thrust_acceleration':'OBJECT1_THRUST_ACCELERATION',
    't_sedr':'OBJECT1_SEDR',
    't_x':'OBJECT1_X',
    't_y':'OBJECT1_Y',
    't_z':'OBJECT1_Z',
    't_x_dot':'OBJECT1_X_DOT',
    't_y_dot':'OBJECT1_Y_DOT',
    't_z_dot':'OBJECT1_Z_DOT',
    't_cr_r':'OBJECT1_CR_R',
    't_ct_r':'OBJECT1_CT_R',
    't_ct_t':'OBJECT1_CT_T',
    't_cn_r':'OBJECT1_CN_R',
    't_cn_t':'OBJECT1_CN_T',
    't_cn_n':'OBJECT1_CN_N',
    't_crdot_r':'OBJECT1_CRDOT_R',
    't_crdot_t':'OBJECT1_CRDOT_T',
    't_crdot_n':'OBJECT1_CRDOT_N',
    't_crdot_rdot':'OBJECT1_CRDOT_RDOT',
    't_ctdot_r':'OBJECT1_CTDOT_R',
    't_ctdot_t':'OBJECT1_CTDOT_T',
    't_ctdot_n':'OBJECT1_CTDOT_N',
    't_ctdot_rdot':'OBJECT1_CTDOT_RDOT',
    't_ctdot_tdot':'OBJECT1_CTDOT_TDOT',
    't_cndot_r':'OBJECT1_CNDOT_R',
    't_cndot_t':'OBJECT1_CNDOT_T',
    't_cndot_n':'OBJECT1_CNDOT_N',
    't_cndot_rdot':'OBJECT1_CNDOT_RDOT',
    't_cndot_tdot':'OBJECT1_CNDOT_TDOT',
    't_cndot_ndot':'OBJECT1_CNDOT_NDOT',
    't_cdrg_r':'OBJECT1_CDRG_R',
    't_cdrg_t':'OBJECT1_CDRG_T',
    't_cdrg_n':'OBJECT1_CDRG_N',
    't_cdrg_rdot':'OBJECT1_CDRG_RDOT',
    't_cdrg_tdot':'OBJECT1_CDRG_TDOT',
    't_cdrg_ndot':'OBJECT1_CDRG_NDOT',
    't_cdrg_drg':'OBJECT1_CDRG_DRG',
    't_csrp_r':'OBJECT1_CSRP_R',
    't_csrp_t':'OBJECT1_CSRP_T',
    't_csrp_n':'OBJECT1_CSRP_N',
    't_csrp_rdot':'OBJECT1_CSRP_RDOT',
    't_csrp_tdot':'OBJECT1_CSRP_TDOT',
    't_csrp_ndot':'OBJECT1_CSRP_NDOT',
    't_csrp_drg':'OBJECT1_CSRP_DRG',
    't_csrp_srp':'OBJECT1_CSRP_SRP',
    't_cthr_r':'OBJECT1_CTHR_R',
    't_cthr_t':'OBJECT1_CTHR_T',
    't_cthr_n':'OBJECT1_CTHR_N',
    't_cthr_rdot':'OBJECT1_CTHR_RDOT',
    't_cthr_tdot':'OBJECT1_CTHR_TDOT',
    't_cthr_ndot':'OBJECT1_CTHR_NDOT',
    't_cthr_drg':'OBJECT1_CTHR_DRG',
    't_cthr_srp':'OBJECT1_CTHR_SRP',
    't_cthr_thr':'OBJECT1_CTHR_THR',
    'c_object_designator':'OBJECT2_OBJECT_DESIGNATOR',
    'c_catalog_name':'OBJECT2_CATALOG_NAME',
    'c_object_name':'OBJECT2_OBJECT_NAME',
    'c_international_designator':'OBJECT2_INTERNATIONAL_DESIGNATOR',
    'c_object_type':'OBJECT2_OBJECT_TYPE',
    'c_ephemeris_name':'OBJECT2_EPHEMERIS_NAME',
    'c_covariance_method':'OBJECT2_COVARIANCE_METHOD',
    'c_maneuverable':'OBJECT2_MANEUVERABLE',
    'c_orbit_center':'OBJECT2_ORBIT_CENTER',
    'c_ref_frame':'OBJECT2_REF_FRAME',
    'c_gravity_model':'OBJECT2_GRAVITY_MODEL',
    'c_atmospheric_model':'OBJECT2_ATMOSPHERIC_MODEL',
    'c_n_body_perturbations':'OBJECT2_N_BODY_PERTURBATIONS',
    'c_solar_rad_pressure':'OBJECT2_SOLAR_RAD_PRESSURE',
    'c_earth_tides':'OBJECT2_EARTH_TIDES',
    'c_intrack_thrust':'OBJECT2_INTRACK_THRUST',
    'c_time_lastob_start':'OBJECT2_TIME_LASTOB_START',
    'c_time_lastob_end':'OBJECT2_TIME_LASTOB_END',
    'c_recommended_od_span':'OBJECT2_RECOMMENDED_OD_SPAN',
    'c_actual_od_span':'OBJECT2_ACTUAL_OD_SPAN',
    'c_obs_available':'OBJECT2_OBS_AVAILABLE',
    'c_obs_used':'OBJECT2_OBS_USED',
    'c_tracks_available':'OBJECT2_TRACKS_AVAILABLE',
    'c_tracks_used':'OBJECT2_TRACKS_USED',
    'c_residuals_accepted':'OBJECT2_RESIDUALS_ACCEPTED',
    'c_weighted_rms':'OBJECT2_WEIGHTED_RMS',
    'c_area_pc':'OBJECT2_AREA_PC',
    'c_area_drg':'OBJECT2_AREA_DRG',
    'c_area_srg':'OBJECT2_AREA_SRP',
    'c_mass':'OBJECT2_MASS',
    'c_cd_area_over_mass':'OBJECT2_CD_AREA_OVER_MASS',
    'c_cr_area_over_mass':'OBJECT2_CR_AREA_OVER_MASS',
    'c_thrust_acceleration':'OBJECT2_THRUST_ACCELERATION',
    'c_sedr':'OBJECT2_SEDR',
    'c_x':'OBJECT2_X',
    'c_y':'OBJECT2_Y',
    'c_z':'OBJECT2_Z',
    'c_x_dot':'OBJECT2_X_DOT',
    'c_y_dot':'OBJECT2_Y_DOT',
    'c_z_dot':'OBJECT2_Z_DOT',
    'c_cr_r':'OBJECT2_CR_R',
    'c_ct_r':'OBJECT2_CT_R',
    'c_ct_t':'OBJECT2_CT_T',
    'c_cn_r':'OBJECT2_CN_R',
    'c_cn_t':'OBJECT2_CN_T',
    'c_cn_n':'OBJECT2_CN_N',
    'c_crdot_r':'OBJECT2_CRDOT_R',
    'c_crdot_t':'OBJECT2_CRDOT_T',
    'c_crdot_n':'OBJECT2_CRDOT_N',
    'c_crdot_rdot':'OBJECT2_CRDOT_RDOT',
    'c_ctdot_r':'OBJECT2_CTDOT_R',
    'c_ctdot_t':'OBJECT2_CTDOT_T',
    'c_ctdot_n':'OBJECT2_CTDOT_N',
    'c_ctdot_rdot':'OBJECT2_CTDOT_RDOT',
    'c_ctdot_tdot':'OBJECT2_CTDOT_TDOT',
    'c_cndot_r':'OBJECT2_CNDOT_R',
    'c_cndot_t':'OBJECT2_CNDOT_T',
    'c_cndot_n':'OBJECT2_CNDOT_N',
    'c_cndot_rdot':'OBJECT2_CNDOT_RDOT',
    'c_cndot_tdot':'OBJECT2_CNDOT_TDOT',
    'c_cndot_ndot':'OBJECT2_CNDOT_NDOT',
    'c_cdrg_r':'OBJECT2_CDRG_R',
    'c_cdrg_t':'OBJECT2_CDRG_T',
    'c_cdrg_n':'OBJECT2_CDRG_N',
    'c_cdrg_rdot':'OBJECT2_CDRG_RDOT',
    'c_cdrg_tdot':'OBJECT2_CDRG_TDOT',
    'c_cdrg_ndot':'OBJECT2_CDRG_NDOT',
    'c_cdrg_drg':'OBJECT2_CDRG_DRG',
    'c_csrp_r':'OBJECT2_CSRP_R',
    'c_csrp_t':'OBJECT2_CSRP_T',
    'c_csrp_n':'OBJECT2_CSRP_N',
    'c_csrp_rdot':'OBJECT2_CSRP_RDOT',
    'c_csrp_tdot':'OBJECT2_CSRP_TDOT',
    'c_csrp_ndot':'OBJECT2_CSRP_NDOT',
    'c_csrp_drg':'OBJECT2_CSRP_DRG',
    'c_csrp_srp':'OBJECT2_CSRP_SRP',
    'c_cthr_r':'OBJECT2_CTHR_R',
    'c_cthr_t':'OBJECT2_CTHR_T',
    'c_cthr_n':'OBJECT2_CTHR_N',
    'c_cthr_rdot':'OBJECT2_CTHR_RDOT',
    'c_cthr_tdot':'OBJECT2_CTHR_TDOT',
    'c_cthr_ndot':'OBJECT2_CTHR_NDOT',
    'c_cthr_drg':'OBJECT2_CTHR_DRG',
    'c_cthr_srp':'OBJECT2_CTHR_SRP',
    'c_cthr_thr':'OBJECT2_CTHR_THR'
}

# Declare global variable containing all keys within every cluster in a CDM.
cdm_clusters = {
    'header':
        ['CCSDS_CDM_VERS', 'CREATION_DATE', 'ORIGINATOR', 'MESSAGE_FOR', 
        'MESSAGE_ID'],

    'relative_metadata': 
        ['TCA', 'MISS_DISTANCE', 'RELATIVE_SPEED', 'RELATIVE_POSITION_R', 
        'RELATIVE_POSITION_T', 'RELATIVE_POSITION_N', 'RELATIVE_VELOCITY_R', 
        'RELATIVE_VELOCITY_T', 'RELATIVE_VELOCITY_N', 'START_SCREEN_PERIOD', 
        'STOP_SCREEN_PERIOD', 'SCREEN_VOLUME_FRAME', 'SCREEN_VOLUME_SHAPE', 
        'SCREEN_VOLUME_X', 'SCREEN_VOLUME_Y', 'SCREEN_VOLUME_Z', 
        'SCREEN_ENTRY_TIME', 'SCREEN_EXIT_TIME', 'COLLISION_PROBABILITY', 
        'COLLISION_PROBABILITY_METHOD'],

    'metadata':
        ['OBJECT', 'OBJECT_DESIGNATOR', 'CATALOG_NAME', 'OBJECT_NAME', 
        'INTERNATIONAL_DESIGNATOR', 'OBJECT_TYPE', 'OPERATOR_CONTACT_POSITION', 
        'OPERATOR_ORGANIZATION', 'OPERATOR_PHONE', 'OPERATOR_EMAIL', 
        'EPHEMERIS_NAME', 'COVARIANCE_METHOD', 'MANEUVERABLE', 'ORBIT_CENTER', 
        'REF_FRAME', 'GRAVITY_MODEL', 'ATMOSPHERIC_MODEL', 
        'N_BODY_PERTURBATIONS', 'SOLAR_RAD_PRESSURE', 'EARTH_TIDES', 
        'INTRACK_THRUST'],

    'data_od': 
        ['TIME_LASTOB_START', 'TIME_LASTOB_END', 'RECOMMENDED_OD_SPAN', 
        'ACTUAL_OD_SPAN', 'OBS_AVAILABLE', 'OBS_USED', 'TRACKS_AVAILABLE', 
        'TRACKS_USED', 'RESIDUALS_ACCEPTED', 'WEIGHTED_RMS', 'AREA_PC', 
        'AREA_DRG', 'AREA_SRP', 'MASS', 'CD_AREA_OVER_MASS', 
        'CR_AREA_OVER_MASS', 'THRUST_ACCELERATION', 'SEDR'],

    'data_state': 
        ['X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT'],

    'data_covariance': 
        ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 
        'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 
        'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 
        'CNDOT_TDOT', 'CNDOT_NDOT', 'CDRG_R', 'CDRG_T', 'CDRG_N', 'CDRG_RDOT', 
        'CDRG_TDOT', 'CDRG_NDOT', 'CDRG_DRG', 'CSRP_R', 'CSRP_T', 'CSRP_N', 
        'CSRP_RDOT', 'CSRP_TDOT', 'CSRP_NDOT', 'CSRP_DRG', 'CSRP_SRP', 'CTHR_R', 
        'CTHR_T', 'CTHR_N', 'CTHR_RDOT', 'CTHR_TDOT', 'CTHR_NDOT', 'CTHR_DRG', 
        'CTHR_SRP', 'CTHR_THR']
    }

cdm_clusters_obligatory = {
    'header_obligatory':
        ['CCSDS_CDM_VERS', 'CREATION_DATE', 'ORIGINATOR', 'MESSAGE_ID'],

    'relative_metadata_obligatory': 
        ['TCA', 'MISS_DISTANCE'],

    'metadata_obligatory':
        ['OBJECT', 'OBJECT_DESIGNATOR', 'CATALOG_NAME', 'OBJECT_NAME', 
        'INTERNATIONAL_DESIGNATOR', 'EPHEMERIS_NAME', 'COVARIANCE_METHOD', 
        'MANEUVERABLE', 'REF_FRAME'],

    'data_od_obligatory': 
        [],

    'data_state_obligatory': 
        ['X', 'Y', 'Z', 'X_DOT', 'Y_DOT', 'Z_DOT'],

    'data_covariance_obligatory': 
        ['CR_R', 'CT_R', 'CT_T', 'CN_R', 'CN_T', 'CN_N', 'CRDOT_R', 'CRDOT_T', 
        'CRDOT_N', 'CRDOT_RDOT', 'CTDOT_R', 'CTDOT_T', 'CTDOT_N', 'CTDOT_RDOT', 
        'CTDOT_TDOT', 'CNDOT_R', 'CNDOT_T', 'CNDOT_N', 'CNDOT_RDOT', 
        'CNDOT_TDOT', 'CNDOT_NDOT']
    }

def cluster_from_key(key:str, obligatory:bool = False) -> str:
    """Get the cluster to which a key belongs to.

    Args:
        key (str): CDM key to check
        obligatory(bool, optional): Flag to determine what cluster shall be used
        to retrieve the name given the key. Defaults to False

    Raises:
        ValueError: Key does not belong to any of the clusters.

    Returns:
        str: Cluster of data the key belongs to.
    """

    # Set output initially as None
    output = None

    # Get relevant data cluster
    clusters = cdm_clusters_obligatory if obligatory else cdm_clusters

    # Iterate over all cluster and their keys.
    for cluster, keys in clusters.items():

        # Skip obligatory clusters.
        if 'obligatory' in cluster: continue

        if key in keys: 
            output = cluster 
            break

    # If output continues to be None, raise an error to report an invalid key.
    # Return the output otherwise.
    if output is None:
        raise ValueError(f'Invalid key ({key}). ' + \
                         f'It does not belong to any data cluster.')
    else:
        return cluster




#%%
# Features clustered by group
features_groups = {
    "targets": [
        "risk", "max_risk_estimate", "max_risk_scaling"
        ],
    "ids": [
        "event_id", "mission_id"
        ],
    "conjunction": [
        "time_to_tca", "c_object_type", "miss_distance", "mahalanobis_distance", 
        "geocentric_latitude", "azimuth", "elevation",
        "F10", "F3M", "SSN", "AP"
        ],
    "relative_state": [
        "relative_position_r", "relative_position_t", "relative_position_n",
        "relative_velocity_r", "relative_velocity_t", "relative_velocity_n",
        "relative_speed"
        ],
    "objects": {
        "coefficients": [
            "cd_area_over_mass", "cr_area_over_mass"
        ],
        "covariance": [
            "ct_r", "cn_r", "cn_t",
            "crdot_r", "crdot_t", "crdot_n",
            "ctdot_r", "ctdot_t", "ctdot_n",
            "cndot_r", "cndot_t", "cndot_n",
            "ctdot_rdot", "cndot_rdot", "cndot_tdot",
            "sigma_r", "sigma_t", "sigma_n",
            "sigma_rdot", "sigma_tdot", "sigma_ndot" 
        ],
        "orb_elements": [
            "h_apo", "h_per", "j2k_ecc", "j2k_inc", "j2k_sma"
        ],
        "miscellaneous": [
            "span", "actual_od_span", "recommended_od_span", "obs_available", "obs_used",
            "time_lastob_end", "time_lastob_start", "position_covariance_det", 
            "rcs_estimate", "sedr", "residuals_accepted", "weighted_rms"
        ]
    }
}

#%%
# Categorical input features
features =     {"event_id":                   	{'input': None, 'compulsory':True, 'continuous': False, 'variable': False, 'independent': False, 'cluster':'ids'},
                "mission_id":                 	{'input': None, 'compulsory':True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'ids'},
    
                "risk":                       	{'input': False, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'targets'},
                "max_risk_estimate":          	{'input': False, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'targets'},
                "max_risk_scaling":           	{'input': None, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'targets'},

                "time_to_tca":                	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "miss_distance":              	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'conjunction'},
                "c_object_type":              	{'input': True, 'compulsory':True, 'continuous': False, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "geocentric_latitude":        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "azimuth":                    	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "elevation":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "mahalanobis_distance":       	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "F10":                        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "F3M":                        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "SSN":                        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "AP":                         	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},

                "time_to_cdm":                  {'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "cdms_to_tca":                  {'input': False, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},

                "relative_speed":             	{'input': True, 'compulsory':True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'relative_state'},
                "relative_velocity_r":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_velocity_t":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_velocity_n":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_r":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_t":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_n":        	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},

                "t_cd_area_over_mass":        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "t_cr_area_over_mass":        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "c_cd_area_over_mass":        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "c_cr_area_over_mass":        	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},

                "t_j2k_ecc":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "t_j2k_inc":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "t_j2k_sma":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "t_h_apo":                    	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "t_h_per":                    	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "c_j2k_sma":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "c_j2k_ecc":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "c_j2k_inc":                  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "c_h_apo":                    	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "c_h_per":                    	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},

                "t_ct_r":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cn_r":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cn_t":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_crdot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_crdot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_crdot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_ctdot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cndot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cndot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_tdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ct_r":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cn_r":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cn_t":                     	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_crdot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_crdot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_crdot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_ctdot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cndot_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cndot_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_tdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_sigma_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_r":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_t":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_n":                  	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_sigma_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_rdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_tdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_sigma_tdot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_ndot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_ndot":               	{'input': True, 'compulsory':True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},

                "t_time_lastob_start":        	{'input': True, 'compulsory':False, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "t_time_lastob_end":          	{'input': True, 'compulsory':False, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "t_recommended_od_span":      	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_actual_od_span":           	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_obs_available":            	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_obs_used":                 	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "t_residuals_accepted":       	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_weighted_rms":             	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_rcs_estimate":             	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "t_sedr":                     	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_time_lastob_start":        	{'input': True, 'compulsory':False, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "c_time_lastob_end":          	{'input': True, 'compulsory':False, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "c_recommended_od_span":      	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_actual_od_span":           	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_obs_available":            	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_obs_used":                 	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_residuals_accepted":       	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_weighted_rms":             	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_rcs_estimate":             	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_sedr":                     	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_span":                     	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "c_span":                     	{'input': True, 'compulsory':False, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "t_position_covariance_det":  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_position_covariance_det":  	{'input': True, 'compulsory':False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'}}

#%%
def get_features(only_names:bool = True, **kwargs):

    if not isinstance(kwargs, dict): return None

    output = {}
    for feature, clusters in features.items():
        exclude = False

        # Run through all filtering criterias passed with kwargs
        for key, value in kwargs.items():

            # If filters passed through kwargs are not a list, convert value as 
            # a list.
            if not isinstance(value, list): value = [value]

            # If a criteria 'key' has multiple conditions, check all parameters 
            # from feature match at least one of those conditions.
            exclude = True if clusters[key] not in value else exclude

        if not exclude: output[feature] = clusters

    return list(dict.keys(output)) if only_names else output