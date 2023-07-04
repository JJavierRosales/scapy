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
features =     {"risk":                       	{'input': False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'targets'},
                "max_risk_estimate":          	{'input': False, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'targets'},
                "max_risk_scaling":           	{'input': None, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'targets'},

                "event_id":                   	{'input': None, 'continuous': False, 'variable': False, 'independent': False, 'cluster':'ids'},
                "mission_id":                 	{'input': None, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'ids'},

                "time_to_tca":                	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "miss_distance":              	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "c_object_type":              	{'input': True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'conjunction'},
                "geocentric_latitude":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "azimuth":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "elevation":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "mahalanobis_distance":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "F10":                        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "F3M":                        	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'conjunction'},
                "SSN":                        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},
                "AP":                         	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'conjunction'},

                "relative_speed":             	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'relative_state'},
                "relative_velocity_r":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_velocity_t":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_velocity_n":        	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_r":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_t":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},
                "relative_position_n":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'relative_state'},

                "t_cd_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "t_cr_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "c_cd_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},
                "c_cr_area_over_mass":        	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'coefficients'},

                "t_j2k_ecc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "t_j2k_inc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "t_j2k_sma":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "t_h_apo":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "t_h_per":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "c_j2k_sma":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "c_j2k_ecc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "c_j2k_inc":                  	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'orb_elements'},
                "c_h_apo":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},
                "c_h_per":                    	{'input': True, 'continuous': True, 'variable': False, 'independent': False, 'cluster': 'orb_elements'},

                "t_ct_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cn_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cn_t":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_crdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_crdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_crdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_ctdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_ctdot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cndot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_cndot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_cndot_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ct_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cn_r":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cn_t":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_crdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_crdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_crdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_ctdot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_ctdot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cndot_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_cndot_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_cndot_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_sigma_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_r":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_t":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_n":                  	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "t_sigma_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_rdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'covariance'},
                "c_sigma_tdot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "t_sigma_ndot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},
                "c_sigma_ndot":               	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'covariance'},

                "t_time_lastob_start":        	{'input': True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "t_time_lastob_end":          	{'input': True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "t_recommended_od_span":      	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_actual_od_span":           	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_obs_available":            	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_obs_used":                 	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "t_residuals_accepted":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_weighted_rms":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_rcs_estimate":             	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "t_sedr":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_time_lastob_start":        	{'input': True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "c_time_lastob_end":          	{'input': True, 'continuous': False, 'variable': False, 'independent': False, 'cluster': 'miscellaneous'},
                "c_recommended_od_span":      	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_actual_od_span":           	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_obs_available":            	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_obs_used":                 	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_residuals_accepted":       	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_weighted_rms":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_rcs_estimate":             	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "c_sedr":                     	{'input': True, 'continuous': True, 'variable': True, 'independent': True, 'cluster': 'miscellaneous'},
                "t_span":                     	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "c_span":                     	{'input': True, 'continuous': True, 'variable': False, 'independent': True, 'cluster': 'miscellaneous'},
                "t_position_covariance_det":  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'},
                "c_position_covariance_det":  	{'input': True, 'continuous': True, 'variable': True, 'independent': False, 'cluster': 'miscellaneous'}}

#%%
def get_features(only_names:bool = True, **kwargs):

    if not isinstance(kwargs, dict): return None

    output = {}
    for feature, clusters in features.items():
        exclude = False
        for key, value in kwargs.items():
            exclude = True if clusters[key] != value else exclude

        if not exclude: output[feature] = clusters

    if only_names:
        return list(dict.keys(output))
    else:
        return output