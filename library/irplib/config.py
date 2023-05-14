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
features =     {"event_id":                   	{'input': None, 'continuous': False, 'variable': False, 'independent': False},
                "time_to_tca":                	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "mission_id":                 	{'input': None, 'continuous': False, 'variable': False, 'independent': False},
                "risk":                       	{'input': False, 'continuous': True, 'variable': True, 'independent': False},
                "max_risk_estimate":          	{'input': False, 'continuous': True, 'variable': True, 'independent': False},
                "max_risk_scaling":           	{'input': None, 'continuous': True, 'variable': False, 'independent': False},
                "miss_distance":              	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "relative_speed":             	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "relative_position_r":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "relative_position_t":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "relative_position_n":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "relative_velocity_r":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "relative_velocity_t":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "relative_velocity_n":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_time_lastob_start":        	{'input': True, 'continuous': False, 'variable': False, 'independent': False},
                "t_time_lastob_end":          	{'input': True, 'continuous': False, 'variable': False, 'independent': False},
                "t_recommended_od_span":      	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_actual_od_span":           	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_obs_available":            	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_obs_used":                 	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_residuals_accepted":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_weighted_rms":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_rcs_estimate":             	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_cd_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_cr_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_sedr":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_j2k_sma":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "t_j2k_ecc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_j2k_inc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_ct_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_cn_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_cn_t":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_crdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_crdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_crdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_ctdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_ctdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_ctdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_ctdot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_cndot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_cndot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_cndot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_cndot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_cndot_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_object_type":              	{'input': True, 'continuous': False, 'variable': False, 'independent': False},
                "c_time_lastob_start":        	{'input': True, 'continuous': False, 'variable': False, 'independent': False},
                "c_time_lastob_end":          	{'input': True, 'continuous': False, 'variable': False, 'independent': False},
                "c_recommended_od_span":      	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_actual_od_span":           	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_obs_available":            	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_obs_used":                 	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_residuals_accepted":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_weighted_rms":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_rcs_estimate":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cd_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cr_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_sedr":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_j2k_sma":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "c_j2k_ecc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "c_j2k_inc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "c_ct_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cn_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_cn_t":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_crdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_crdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_crdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_ctdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_ctdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_ctdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_ctdot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cndot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_cndot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_cndot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cndot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_cndot_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_span":                     	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "c_span":                     	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_h_apo":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "t_h_per":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "c_h_apo":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "c_h_per":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False},
                "geocentric_latitude":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "azimuth":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "elevation":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "mahalanobis_distance":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_position_covariance_det":  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_position_covariance_det":  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_sigma_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_sigma_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_sigma_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_sigma_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_sigma_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_sigma_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "t_sigma_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_sigma_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_sigma_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "c_sigma_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "t_sigma_ndot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "c_sigma_ndot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False},
                "F10":                        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "F3M":                        	{'input': True, 'continuous': True, 'variable': False, 'independent': True},
                "SSN":                        	{'input': True, 'continuous': True, 'variable': True, 'independent': True},
                "AP":                         	{'input': True, 'continuous': True, 'variable': True, 'independent': True}}

#%%
def get_features(**kwargs):

    if not isinstance(kwargs, dict): return None


    output = {}
    for feature, clusters in features.items():
        exclude = False
        for key, value in kwargs.items():
            exclude = True if clusters[key] != value else exclude

        if not exclude: output[feature] = clusters

    return output